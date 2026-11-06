#!/usr/bin/env bash
# 训练冒烟的全部逻辑：校验映射表 → 装栈 → 取模型 → 推并行策略 → 起训练 → 出摘要。
#
# 为什么不在 yaml 里：这段有 400 行、每个分支都是踩过的坑（见各段注释）。
# 写在 `run: |` 里既不能本地跑、也过不了 shellcheck，改一行只能推上去等一小时。
#
# 子命令（对应 workflow 的步骤）：
#   prepare  ②  校验 model_type / 启动脚本 / 依赖脚本，映射表漂移
#   probe    ②b NPU 与栈自检（每张卡真算一次 matmul）
#   install  ③④ pip 装 torchturbo 与模型依赖（torch 归属校验）
#   run      ⑤  取模型 → 推并行策略 → 起训练 → 落事实
#   report   ⑥  渲染 Job Summary 与注解
#   lane     prepare → install → run → report，按 E2E_PROFILES 逐个跑
#
# 事实写进 $RESULT_JSON 而不是 step outputs：step 失败时 outputs 是空的，
# 而失败的那次恰恰是唯一会被人翻出来看的。

set -uo pipefail

CI_SCRIPTS="${CI_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RESULT_JSON="${RESULT_JSON:-/tmp/ci-result.json}"
GITHUB_OUTPUT="${GITHUB_OUTPUT:-/dev/null}"
MAP="${MAP:-.github/workflows/model_map.json}"

# set -u turns any missing input into a crash halfway through a 20 min job.
# Default everything the functions read, in one place.
: "${MODEL_TYPE:=}"; : "${FRAMEWORK:=lf}"; : "${RUNNER_LABEL:=}"
: "${IMAGE:=}"; : "${LAYERS:=0}"
: "${MODEL_HINT:=}"; : "${REPO_MAP:=}"; : "${REPO_IN:=}"
: "${SP_IN:=auto}"; : "${EP_IN:=auto}"; : "${SEQ_LEN:=1024}"
: "${FINETUNING_TYPE:=full}"; : "${DATASET_IN:=gsm8k}"
# DATASET = what the launchers read; DATASET_IN = the workflow input name.
: "${DATASET:=${DATASET_IN}}"
: "${DIAG:=false}"; : "${FETCH_TIMEOUT:=600}"; : "${STRICT:=false}"
: "${INSTALL_SCRIPT:=}"; : "${TRAIN_SCRIPT:=}"; : "${E2E_PROFILES:=[]}"

result() { python3 "$CI_SCRIPTS/report_result.py" "$@"; }

# 每步都落一条 phase，摘要里按执行顺序列表。
phase_run() {                       # phase_run <名字> <命令...>
  local name="$1"; shift
  local start rc
  start=$(date +%s)
  "$@"
  rc=$?
  result phase --file "$RESULT_JSON" --name "$name" \
    --seconds "$(( $(date +%s) - start ))" --rc "$rc"
  echo "::notice::${name} 用时 $(( $(date +%s) - start ))s（rc=${rc}）"
  return "$rc"
}

# ---------------------------------------------------------------- prepare
# 先确认这个 model_type 真能跑，再去装依赖；否则白等几十分钟。
# 字段：run_dir lf swift hf_repo e2e（顺序由 report_result.py 的 _MAP_FIELDS 决定）
cmd_prepare() {
  result init --file "$RESULT_JSON" \
    kind=train model_type="$MODEL_TYPE" framework="$FRAMEWORK" \
    runner_label="$RUNNER_LABEL" image="$IMAGE" \
    layers="$LAYERS" dataset="$DATASET"

  [ -f "$MAP" ] || { echo "::error::缺少 $MAP"; return 1; }

  # 映射表是 JSON，用 awk 取字段就得现写一个 JSON 解析器 —— 所以让
  # report_result.py 读，shell 只负责把值读进变量。字段顺序见 _MAP_FIELDS。
  local row
  if ! row=$(result map-row --map "$MAP" --model-type "$MODEL_TYPE"); then
    echo "::error::model_type '$MODEL_TYPE' 不在 $MAP 里；可选项："
    result map-keys --map "$MAP" | sed 's/^/             /'
    return 1
  fi
  # shellcheck disable=SC2086
  set -- $row
  local run_dir=$1 launchers
  case "$FRAMEWORK" in
    lf)    launchers=$2 ;;
    swift) launchers=$3 ;;
    *)     echo "::error::未知 framework '$FRAMEWORK'（只支持 lf / swift）"; return 1 ;;
  esac
  local repo=${4:--}
  echo "model_type=$MODEL_TYPE framework=$FRAMEWORK run_dir=$run_dir launchers=$launchers repo=$repo"

  if [ "$run_dir" = "-" ] || [ ! -d "scripts/runs/$run_dir" ]; then
    echo "::error::$MODEL_TYPE 没有可用的 run_dir：scripts/runs/$run_dir"
    ls -1 scripts/runs | sed 's/^/             /'
    return 1
  fi
  if [ "$launchers" = "-" ] || [ -z "$launchers" ]; then
    echo "::error::$MODEL_TYPE 在 $FRAMEWORK 下没有启动脚本（映射表里 ${FRAMEWORK} 字段为 '-'）"
    echo "            同一行：run_dir=$run_dir lf=[$2] swift=[$3]"
    return 1
  fi

  local train="" l
  for l in $(echo "$launchers" | tr ',' ' '); do
    [ -f "scripts/runs/$run_dir/$l" ] && { train="scripts/runs/$run_dir/$l"; break; }
  done
  if [ -z "$train" ]; then
    echo "::error::$MODEL_TYPE/$FRAMEWORK 的 launchers 一个都不存在：$launchers"
    ls -1 "scripts/runs/$run_dir" | sed 's/^/             /'
    return 1
  fi

  # 依赖安装脚本：优先按框架选，退回通用的 install_lf_deps.sh
  local install="" c
  for c in "scripts/runs/$run_dir/install_${FRAMEWORK}_deps.sh" \
           "scripts/runs/$run_dir/install_lf_deps.sh"; do
    [ -f "$c" ] && { install="$c"; break; }
  done
  if [ -z "$install" ]; then
    echo "::error::$MODEL_TYPE 没有可用的依赖安装脚本（install_*.sh 都不存在）"
    ls -1 "scripts/runs/$run_dir" | sed 's/^/             /'
    return 1
  fi

  echo "校验通过：train=$train install=$install"
  result set --file "$RESULT_JSON" \
    run_dir="$run_dir" install_script="$install" train_script="$train"
  {
    echo "run_dir=$run_dir"
    echo "install_script=$install"
    echo "train_script=$train"
    echo "hf_repo=$repo"
  } >> "$GITHUB_OUTPUT"

  # 注册表漂移：映射表必须覆盖代码里全部 @register_model 的名字。
  # 与 L0 的 check_static 同一套判据，只是这里不装依赖也能跑（只用标准库）。
  python3 "$CI_SCRIPTS/check_static.py" --only map
}

# ---------------------------------------------------------------- probe
# 花 20 s 发现「没有 NPU」，比装完两套依赖再发现省一整轮超时，
# 而且那时的报错长得像训练 bug。env_pod.py verify 会真的在每张卡上算一次 matmul，
# 这是唯一能区分「没设备」和「有设备但栈坏了」的检查。
cmd_probe() {
  # 在任何 pip 之前记录；install 会拿它比对。
  python3 "$CI_SCRIPTS/env_pod.py" capture --file /tmp/torch-before.json

  ls /dev/davinci* >/dev/null 2>&1 \
    || { echo "::error::no /dev/davinci*, NPUs not mapped into the pod"; return 1; }
  echo "davinci count: $(ls /dev/davinci* | wc -l)"

  phase_run '②b NPU 自检' python3 "$CI_SCRIPTS/env_pod.py" verify
}

# ---------------------------------------------------------------- install
# 两个安装点在踩同一个坑：pip 解依赖时会用 PyPI 的 torch 覆盖自建的 NPU wheel，
# 而且整个过程一声不响。所以每次装完都回头验一遍 torch 是从哪来的。
pip_step() {                        # pip_step <名字> <安装脚本或空>
  local name="$1" script="${2:-}" log start rc
  log="/tmp/$(echo "$name" | tr -cd '[:alnum:]' | head -c 20).log"
  start=$(date +%s)

  python -m pip config list 2>/dev/null || true
  env | grep -iE '^(PIP_|.*_PROXY=)' || true

  if [ -n "$script" ]; then
    test -f "$script" || { echo "::error::缺少 $script"; return 1; }
    cat "$script"
    bash "$script" 2>&1 | tee "$log"
  else
    # --no-build-isolation：否则 pip 会为了跑 setuptools 额外下一个完整 torch。
    # 不加 -q：Collecting/Downloading 行是「到底有没有重新下载」的直接证据。
    python -m pip install --no-cache-dir --no-build-isolation -e ".[dev]" 2>&1 | tee "$log"
  fi
  rc=${PIPESTATUS[0]}
  local elapsed=$(( $(date +%s) - start ))

  grep -E '^(Collecting|Downloading|Using cached|Installing collected|Requirement already satisfied)' \
    "$log" | sed 's/ (from .*//' || true

  # 装完实际落了什么版本。故意放在计时区间外，且永不致命。
  python - >/tmp/after.txt 2>&1 <<'PROBE' || true
import importlib
for n in "torch torch_npu numpy triton fla fla_npu transformers llamafactory deepspeed omegaconf pytest".split():
    try:
        print(f"{n}={getattr(importlib.import_module(n), '__version__', 'unknown')}")
    except Exception as e:
        print(f"{n}=MISSING({type(e).__name__})")
PROBE
  cat /tmp/after.txt

  if [ "$rc" -eq 0 ]; then
    python3 "$CI_SCRIPTS/env_pod.py" check --file /tmp/torch-before.json || rc=1
  fi
  result phase --file "$RESULT_JSON" --name "$name" --seconds "$elapsed" --rc "$rc"
  echo "::notice::${name} 耗时 ${elapsed}s（rc=${rc}）"
  return "$rc"
}

cmd_install() {
  pip_step '③ torchturbo `-e .[dev]`' || return 1
  pip_step "④ \`${INSTALL_SCRIPT}\`" "$INSTALL_SCRIPT"
}

# ---------------------------------------------------------------- run
cmd_run() {
  # 卡数直接取 runner 标签的尾缀（-2 / -4 / -8），避免与 yaml 里的 sp_size 不匹配。
  NPUS_PER_NODE="${RUNNER_LABEL##*-}"
  case "$NPUS_PER_NODE" in ''|*[!0-9]*) NPUS_PER_NODE=2 ;; esac
  export NPUS_PER_NODE
  echo "cards=${NPUS_PER_NODE} (来自 ${RUNNER_LABEL})"

  # 卷身份：不同 run 会落到不同的 SFS，缓存命中与否取决于它。
  local uuid
  uuid=$(findmnt -nT /root/.cache -o SOURCE 2>/dev/null | sed -n '1p' | cut -d. -f1)
  echo "sfs_uuid=${uuid:-?}"

  local hf_repo="${REPO_IN:-$REPO_MAP}"
  if [ -z "$MODEL_HINT" ] && [ "$hf_repo" = "-" ]; then
    echo "::error::$MODEL_TYPE 没登记 hf_repo（映射表第 5 列为 '-'），必须显式传 train_model_path"
    return 1
  fi

  # 超时即跳过：网络不可控，拉不动就报告并让 gate 记 skipped-reason，
  # 而不是让整个 PR 以「模型下载失败」收场。
  skip_out() {
    echo "::warning::$1"
    echo "skip_reason=$1" >> "$GITHUB_OUTPUT"
    result set --file "$RESULT_JSON" skip_reason="$1"
    result phase --file "$RESULT_JSON" --name '⑤ 模型获取' --seconds "$2" --rc 0
    [ "${STRICT:-false}" = "true" ] && exit 1
    exit 0
  }

  # ---- 模型获取 ----------------------------------------------------
  # 缓存键带上层数与上游 revision：改了裁剪口径或不钉版本都会静默命中旧权重。
  local marker="repo=${hf_repo} layers=${LAYERS}"
  local cache_out=/tmp/cache.out model="" bytes=0 shards=0 hit=no
  : > "$cache_out"

  if [ -n "$MODEL_HINT" ]; then
    if r=$(python3 "$CI_SCRIPTS/fetch_model.py" probe --dir "$MODEL_HINT"); then
      model=$MODEL_HINT; bytes=${r% *}; shards=${r#* }; hit=yes
      echo "使用显式模型：$model  bytes=$bytes shards=$shards"
    else
      echo "::error::train_model_path 不完整：$MODEL_HINT"; return 1
    fi
  else
    python3 "$CI_SCRIPTS/fetch_model.py" find --repo "$hf_repo" \
      --marker "$marker" --output "$cache_out"
    # find writes uppercase MODEL / BYTES / SHARDS / HIT; the rest of this
    # function works in lowercase. Sourcing alone leaves `model` empty, which
    # silently exports MODEL="" and the launcher falls back to a path that
    # does not exist (HFValidationError on '/models/...' several minutes in).
    # shellcheck disable=SC1090
    . "$cache_out"
    if [ "${HIT:-no}" = yes ]; then
      model="${MODEL:-}"; bytes="${BYTES:-0}"; shards="${SHARDS:-0}"
      [ -n "$model" ] || { echo "::error::fetch_model.py find 报了 HIT 但没给 MODEL"; return 1; }
      echo "缓存命中：$model"
    else
      local fetch_start trc
      fetch_start=$(date +%s)
      if [ -z "${LAYERS:-0}" ] || [ "${LAYERS:-0}" -eq 0 ]; then
        model="/root/.cache/models/$hf_repo"
        echo "::notice::缓存未命中（sfs=${uuid:-?}），下载 $hf_repo"
        timeout "$FETCH_TIMEOUT" python3 "$CI_SCRIPTS/fetch_model.py" fetch \
          --repo "$hf_repo" --dir "$model"
        trc=$?
      else
        model="/root/.cache/models/$(basename "$hf_repo")-L${LAYERS}"
        echo "::notice::缓存未命中（sfs=${uuid:-?}），裁剪 $hf_repo 到 $LAYERS 层"
        timeout "$FETCH_TIMEOUT" python3 "$CI_SCRIPTS/fetch_model.py" trim \
          --repo "$hf_repo" --dir "$model" --layers "$LAYERS"
        trc=$?
      fi
      # 124 = timeout 杀的。
      if [ "$trc" -eq 124 ]; then
        skip_out "model-fetch-timeout(${FETCH_TIMEOUT}s)" "$(( $(date +%s) - fetch_start ))"
      fi
      if [ "$trc" -ne 0 ]; then
        echo "::error::模型获取失败（rc=${trc}）：$hf_repo"
        return 1
      fi
      if ! r=$(python3 "$CI_SCRIPTS/fetch_model.py" probe --dir "$model"); then
        echo "::error::模型不完整：$model"; return 1
      fi
      bytes=${r% *}; shards=${r#* }
      python3 "$CI_SCRIPTS/fetch_model.py" mark --dir "$model" --text "$marker"
    fi
  fi
  echo "model: $model  bytes=$bytes shards=$shards  layers=${LAYERS:-0}"

  # ---- 并行策略 ----------------------------------------------------
  # 仓库里的 yaml 是按满卡写的，不能改（CI 要测的就是仓库里那份），
  # 所以按实际卡数收敛出一份派生配置写到 /tmp，其余字段原样保留。
  test -f "$TRAIN_SCRIPT" || { echo "::error::缺少 $TRAIN_SCRIPT"; return 1; }
  local d src plan_out
  d=$(dirname "$TRAIN_SCRIPT")
  # 取脚本里 TT_CONFIG 的默认表达式，把 $(dirname "$0") / ${SCRIPT_DIR} 展开。
  src=$(sed -n 's/^TT_CONFIG=${TT_CONFIG:-\(.*\)}$/\1/p' "$TRAIN_SCRIPT" | head -1)
  src=$(printf '%s' "$src" | sed -e "s|\$(dirname \"\$0\")|$d|g" \
                                  -e "s|\${SCRIPT_DIR}|$d|g" -e 's|^"||; s|"$||')
  [ -f "$src" ] || src=""
  [ -n "$src" ] || echo "::warning::$TRAIN_SCRIPT 没有配套 yaml，将生成最小配置"

  plan_out=/tmp/plan.out
  : > "$plan_out"
  python3 "$CI_SCRIPTS/plan_parallel.py" \
    --yaml "$src" --cards "$NPUS_PER_NODE" \
    --sp "$SP_IN" --ep "$EP_IN" --layers "${LAYERS:-0}" \
    --out "/tmp/tt_${MODEL_TYPE}.yaml" --output "$plan_out"
  # shellcheck disable=SC1090
  . "$plan_out"
  [ -n "${SP:-}" ] && [ -n "${EP:-}" ] \
    || { echo "::error::并行策略推导失败（见上面的 plan_parallel.py 输出）"; return 1; }
  echo "train script: $TRAIN_SCRIPT"
  echo "tt_config: $TT_CONFIG"

  # ---- 启动 --------------------------------------------------------
  # 启动脚本都是 DATASET=${DATASET:-<各自的默认>}，从外面指定即可。
  export DATASET="$DATASET_IN"
  export MODEL="$model" NPUS_PER_NODE SEQ_LEN FINETUNING_TYPE TT_CONFIG

  # 诊断包：sitecustomize 在每个 python 进程启动时装钩子。
  if [ "${DIAG:-false}" = "true" ]; then
    mkdir -p /tmp/diag
    ln -sfn "$CI_SCRIPTS/diag/sitecustomize.py" /tmp/diag/sitecustomize.py
    # 不能写成 ":/tmp/diag:$PYTHONPATH"：空段等于把 cwd 塞进 sys.path。
    export PYTHONPATH="/tmp/diag${PYTHONPATH:+:$PYTHONPATH}"
    # 单份 HF 导出：裸的 _save_checkpoint 会先把优化器状态写满 overlay。
    export TT_SKIP_CHECKPOINT=1
    echo "diag on (TT_DIAG_CALLS=${TT_DIAG_CALLS:-2})"
  fi

  local train_out=/tmp/train_meta.out elapsed=0 last_loss="" watchdog=no stuck=no
  : > "$train_out"
  bash "$CI_SCRIPTS/supervise_train.sh" \
    --script "$TRAIN_SCRIPT" --log /tmp/train.log --output "$train_out"
  local rc=$?
  # shellcheck disable=SC1090
  . "$train_out"
  elapsed=${ELAPSED:-0}; last_loss=${LAST_LOSS:-}
  watchdog=${WATCHDOG:-no}; stuck=${STUCK:-no}

  echo "::notice::训练 rc=${rc} 用时 ${elapsed}s；hit=${HIT:-?}；${last_loss:-无 loss 记录}"

  # 训练后看盘：子目录也要细分，否则「一份还是两份」只能靠推。
  local out="${OUTPUT_DIR:-/tmp/tblf-sft/$(basename "$model")}"
  df -h "$out" 2>/dev/null | sed 's/^/  df: /'
  du -sh "$out" 2>/dev/null | sed 's/^/  used: /'
  du -sh "$out"/* 2>/dev/null | sed 's/^/    /'
  if [ -s /tmp/tt_traceback.log ]; then
    echo "--- 捕获到的 Python 异常（torchrun 转发时会截断） ---"
    tail -40 /tmp/tt_traceback.log
  fi
  if grep -q '\[ttdiag\]' /tmp/train.log 2>/dev/null; then
    echo "--- loss 诊断（按层 / chunk_loss 入参 / assert_finite） ---"
    grep '\[ttdiag\]' /tmp/train.log | head -400
  fi

  result set --file "$RESULT_JSON" \
    cache_hit="${HIT:-no}" sfs_uuid="${uuid:-?}" \
    shards="${shards:-?}" model_size="$(( ${bytes:-0} / 1048576 )) MiB" \
    sp="$SP" ep="$EP" output_dir="$out" model_path="$model" \
    last_loss="$last_loss" watchdog="$watchdog" \
    stalled="$stuck" skip_reason=""
  result phase --file "$RESULT_JSON" --name '⑤ train' --seconds "${elapsed:-0}" --rc "$rc"
  return "$rc"
}

# ---------------------------------------------------------------- report
cmd_report() {
  result migrate-map --file "$RESULT_JSON" --map "$MAP"
  result report --file "$RESULT_JSON" \
    --train-log /tmp/train.log \
    --traceback /tmp/tt_traceback.log \
    --versions /tmp/after.txt >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
  # 注解是唯一匿名可读的输出通道，关键数字也放这里。
  result notice --file "$RESULT_JSON" --model-type "${MODEL_TYPE:-}" || true
}

# ---------------------------------------------------------------- lane
# One job, N profiles.
# A reusable workflow call expands into *all* of its jobs, and a matrix
# expands once per entry, so selecting profiles with a matrix turns one lane
# into one box per profile. The loop lives here instead: the graph always
# shows exactly one box per lane.
cmd_lane() {
  local rc=0 total=0 skipped=0 per mt layers

  # Callers pass the *_IN spellings (they mirror the workflow input names);
  # the inner functions read the short ones.
  export DATASET="${DATASET_IN:-${DATASET:-gsm8k}}"

  while IFS=$'\t' read -r mt layers; do
    [ -n "$mt" ] || continue
    total=$((total + 1))
    per="/tmp/profile-$total.env"; : > "$per"
    echo "::group::profile $total: $mt (layers=${layers:-0})"
    export MODEL_TYPE="$mt" LAYERS="${layers:-0}"

    # prepare/run talk to $GITHUB_OUTPUT; point it at a temp file so the
    # resolved script paths can be read back per profile.
    if ! GITHUB_OUTPUT="$per" cmd_prepare; then
      rc=1
    else
      # prepare writes lowercase keys (run_dir / install_script / train_script /
      # hf_repo); install and run read the uppercase spellings. In a single
      # workflow job the step outputs bridge the two, here nothing does, so
      # map them explicitly -- empty values are silent (deps not installed,
      # hf_repo='' -> HFValidationError hours later).
      local run_dir install_script train_script hf_repo
      set -a
      # shellcheck disable=SC1090
      . "$per"
      set +a
      RUN_DIR="${run_dir:-}"
      INSTALL_SCRIPT="${install_script:-}"
      TRAIN_SCRIPT="${train_script:-}"
      REPO_MAP="${hf_repo:-}"
      export RUN_DIR INSTALL_SCRIPT TRAIN_SCRIPT REPO_MAP
      if ! cmd_install; then
        rc=1
      else
        # cmd_run exits on a skipped fetch; the subshell keeps the loop alive.
        ( GITHUB_OUTPUT="$per" cmd_run ) || rc=1
        grep -q '^skip_reason=' "$per" && skipped=$((skipped + 1))
      fi
    fi

    cmd_report || true
    echo "::endgroup::"
  done < <(python3 -c '
import json, sys
for p in json.loads(sys.argv[1] or "[]"):
    print("%s\t%s" % (p["model_type"], p.get("layers", 0)))
' "${E2E_PROFILES:-[]}")

  if [ "$total" -eq 0 ]; then
    echo "::notice::no profile selected, nothing to train"
    echo "skip_reason=no profile selected" >> "$GITHUB_OUTPUT"
  elif [ "$skipped" -eq "$total" ]; then
    echo "skip_reason=all $total profile(s) skipped" >> "$GITHUB_OUTPUT"
  fi
  return "$rc"
}

# ---------------------------------------------------------------- 调度
case "${1:-}" in
  prepare) cmd_prepare ;;
  probe)   cmd_probe ;;
  install) cmd_install ;;
  run)     cmd_run ;;
  report)  cmd_report ;;
  lane)    cmd_lane ;;
  *)
    echo "用法：$0 {prepare|probe|install|run|report|lane}" >&2
    exit 2
    ;;
esac
