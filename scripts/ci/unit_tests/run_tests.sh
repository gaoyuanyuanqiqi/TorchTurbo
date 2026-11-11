#!/usr/bin/env bash
# Unit-test job:
#   install - install dependencies for MODE=cpu|npu
#   test    - run one lane: plan -> marker -> collect -> pytest -> report
#
# Test stages:
#   1. plan    - resolve selected files
#   2. marker  - choose the explicit device marker and collection flags
#   3. collect - verify collection without running tests
#   4. pytest  - run the lane and emit JUnit XML
#   5. report  - record timing, test facts, and job output
#
# Environment:
#   MODE               cpu | npu
#   TEST_FILES         JSON array; empty means the whole lane
#   TEST_PATH          path / node id; overrides TEST_FILES
#   PYTEST_ARGS        extra pytest arguments, intentionally word-split
#   TRANSFORMERS_SPEC  optional install-time version constraint
#   TORCH_BEFORE       NPU torch snapshot used after install
#   JUNIT_OUT          JUnit XML output
#   RESULT_JSON        report_result.py facts file
set -uo pipefail

CI_SCRIPTS="${CI_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${MODE:-cpu}"
TEST_FILES="${TEST_FILES:-}"
TEST_PATH="${TEST_PATH:-}"
# "-" requests the whole lane: an empty dispatch input falls back to a default.
if [ "$TEST_PATH" = "-" ]; then
  TEST_PATH=""
fi
PYTEST_ARGS="${PYTEST_ARGS:--v --durations=20}"
JUNIT_OUT="${JUNIT_OUT:-/tmp/pytest.xml}"
RESULT_JSON="${RESULT_JSON:-/tmp/ci-result.json}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-}"
TORCH_BEFORE="${TORCH_BEFORE:-/tmp/torch-before.json}"
LOG="${LOG:-/tmp/turbo.log}"

pip_run() {                         # $1: yes = truncate log, no = append
  local first="$1"; shift
  if [ "$first" = yes ]; then
    python -m pip install "$@" 2>&1 | tee "$LOG"
  else
    python -m pip install "$@" 2>&1 | tee -a "$LOG"
  fi
  return "${PIPESTATUS[0]}"
}

print_versions() {
  python -c "import transformers; print('transformers =', transformers.__version__)" \
    || echo "::warning::transformers is not installed"
}

# ---------------------------------------------------------------- Install
# Install dependencies for the selected lane before the test stages run.
cmd_install() {
  local start rc=0
  start=$(date +%s)
  echo "--- stage: install ---"

  if [ "$MODE" = "npu" ]; then
    # The image already has the custom NPU torch wheel.
    pip_run yes --no-cache-dir --no-build-isolation -e ".[test]" || rc=$?
    print_versions
    if [ "$rc" -eq 0 ]; then
      python3 "$CI_SCRIPTS/common/env_pod.py" check --file "$TORCH_BEFORE" || rc=1
    fi
  else
    # Hosted CPU runners need a usable torch before installing the package.
    pip_run yes torch || rc=$?
    if [ "$rc" -eq 0 ]; then
      pip_run no --no-build-isolation -e ".[test]" || rc=$?
    fi
    if [ "$rc" -eq 0 ] && [ -n "$TRANSFORMERS_SPEC" ]; then
      pip_run no "$TRANSFORMERS_SPEC" || rc=$?
    fi
    print_versions
  fi

  python3 "$CI_SCRIPTS/report/report_result.py" phase --file "$RESULT_JSON" --name 'install' \
    --seconds "$(( $(date +%s) - start ))" --rc "$rc" || true
  echo "::notice::install mode=${MODE} rc=${rc} in $(( $(date +%s) - start ))s"
  return "$rc"
}

case "${1:-test}" in
  install) cmd_install; exit $? ;;
  test) ;;
  *)
    echo "Usage: $0 {install|test}" >&2
    exit 2
    ;;
esac

# Facts file: the workflow initialises it (runner_label / image / kind) before
# calling `install`, so this must not init again - that would rewrite the file
# and drop them along with the install phase. `set` merges instead. Only a local
# run without a facts file needs the full init.
if [ -f "$RESULT_JSON" ]; then
  python3 "$CI_SCRIPTS/report/report_result.py" set --file "$RESULT_JSON" \
    mode="$MODE" test_path="${TEST_PATH:-auto}" pytest_args="$PYTEST_ARGS"
else
  python3 "$CI_SCRIPTS/report/report_result.py" init --file "$RESULT_JSON" \
    kind=pytest model_type="pytest-${MODE}" mode="$MODE" \
    test_path="${TEST_PATH:-auto}" pytest_args="$PYTEST_ARGS"
fi

# ---------------------------------------------------------------- Stage 1: plan
# TEST_PATH overrides the selector; spaces separate paths, except node ids.
echo "--- stage: plan ---"
if [ -n "$TEST_PATH" ]; then
  if [[ "$TEST_PATH" == *::* ]]; then
    printf '%s\n' "$TEST_PATH" > /tmp/tt_pytest_files.txt
  else
    printf '%s\n' "$TEST_PATH" | tr -s '[:space:]' '\n' \
      | sed '/^$/d' > /tmp/tt_pytest_files.txt
  fi
else
python3 - "$TEST_FILES" >/tmp/tt_pytest_files.txt <<'PY'
import glob, json, sys

raw = sys.argv[1]
files = json.loads(raw) if raw.strip() else []
if not files:
    files = sorted(glob.glob("tests/**/test_*.py", recursive=True))
print("\n".join(files))
PY
fi

mapfile -t FILES < /tmp/tt_pytest_files.txt
if [ "${#FILES[@]}" -eq 0 ]; then
  echo "::error::no test files selected (selector and fallback are empty)"
  python3 "$CI_SCRIPTS/report/report_result.py" phase --file "$RESULT_JSON" \
    --name 'pytest' --seconds 0 --rc 1
  exit 1
fi

echo "mode=${MODE} files=${#FILES[@]}"
printf '  %s\n' "${FILES[@]}" | head -30

# ---------------------------------------------------------------- Stage 2: marker
# The lane's device marker always applies, TEST_PATH or not: a mixed file
# (`test_varlen_metadata.py` has cpu and npu tests) must not run its npu half on
# the CPU runner. Asking the wrong lane for a test selects nothing and exits 5.
echo "--- stage: marker ---"
if [ "$MODE" = "npu" ]; then
  MARK=(-m npu)
else
  MARK=(-m cpu)
fi

COLLECT_FLAGS=(--continue-on-collection-errors)

# ---------------------------------------------------------------- Stage 3: collect
# Keep collection errors per-file instead of aborting the whole session.
echo "--- stage: collect ---"
# shellcheck disable=SC2086  # PYTEST_ARGS must be word-split
python -m pytest "${FILES[@]}" "${MARK[@]}" ${PYTEST_ARGS} "${COLLECT_FLAGS[@]}" --collect-only -q 2>&1 \
  | tail -20 || true

# ---------------------------------------------------------------- Stage 4: pytest
# Run the selected lane and write JUnit XML for the report stage.
echo "--- stage: pytest ---"
START=$(date +%s)
set +e
# shellcheck disable=SC2086
python -m pytest "${FILES[@]}" "${MARK[@]}" ${PYTEST_ARGS} "${COLLECT_FLAGS[@]}" \
  --junitxml="$JUNIT_OUT" 2>&1 | tee /tmp/pytest.log
RC=${PIPESTATUS[0]}
ELAPSED=$(( $(date +%s) - START ))

python3 "$CI_SCRIPTS/report/report_result.py" phase --file "$RESULT_JSON" \
  --name "pytest (${MODE})" --seconds "$ELAPSED" --rc "$RC"
echo "::notice::pytest mode=${MODE} rc=${RC} in ${ELAPSED}s"

# ---------------------------------------------------------------- Stage 5: report
# Record phase timing, test facts, and the final job output.
echo "--- stage: report ---"
python3 "$CI_SCRIPTS/report/report_result.py" junit --junit "$JUNIT_OUT" || true

echo "rc=${RC}" >> "${GITHUB_OUTPUT:-/dev/null}"
exit "$RC"
