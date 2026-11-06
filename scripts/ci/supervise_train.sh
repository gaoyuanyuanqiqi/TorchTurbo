#!/bin/bash
# Run a launch script in the background, watch it, and never leave it behind.
#
# Two problems this solves, both observed in real runs:
#
# **A stalled tee.** The launch scripts end in `torchrun ... | tee $LOG_FILE`. When
# the trainer raises, the dataloader's worker pool (forked by `datasets`) survives as
# orphans that still hold the write end of the pipe, so `tee` never sees EOF and the
# script never returns -- one run sat there for 40+ minutes and was killed by the job
# timeout. The scripts belong to the upstream repo, so the supervision lives here.
#
# Detection is by process-tree shape, not by command line: the pool workers are
# forked without exec, so their cmdline is byte-identical to the trainer's and
# `pgrep -f run_lf.py` matches forever. What does change is that the script's direct
# children drop to just `tee` once the trainer is gone, and that state persists.
#
# **A cancelled step.** Neither workflow cancels in progress today, but a timeout or
# a manual cancel sends SIGTERM to this process only. Because the trainer runs in its
# own session (needed so the whole group can be signalled), it would keep running and
# keep holding the NPUs until the pod is recycled. The EXIT/INT/TERM trap tears the
# group down on every path.
#
# Output convention matches fetch_model.py: logs on stderr, `KEY=value` to --output.

set -uo pipefail

LOG=/dev/null
OUT=/dev/stdout
SCRIPT=""
POLL=5
STUCK_LIMIT=3

log() { printf '[supervisor] %s\n' "$*" >&2; }
die() { log "error: $*"; exit 2; }

usage() {
    cat >&2 <<USAGE
usage: supervise_train.sh --script PATH [--log PATH] [--output FILE]
                           [--poll SECONDS] [--stuck-limit N]
USAGE
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --script) SCRIPT=$2; shift 2 ;;
        --log) LOG=$2; shift 2 ;;
        --output) OUT=$2; shift 2 ;;
        --poll) POLL=$2; shift 2 ;;
        --stuck-limit) STUCK_LIMIT=$2; shift 2 ;;
        -h|--help) usage ;;
        *) die "unknown option $1" ;;
    esac
done
[ -n "$SCRIPT" ] || usage
[ -f "$SCRIPT" ] || die "missing launch script: $SCRIPT"

TRAINPID=""
TAILPID=""

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [ -n "$TAILPID" ]; then
        kill "$TAILPID" 2>/dev/null
    fi
    # Signal the whole group: without this, a cancelled step leaves torchrun's
    # workers holding every NPU on the node.
    if [ -n "$TRAINPID" ] && kill -0 "$TRAINPID" 2>/dev/null; then
        log "tearing down process group $TRAINPID"
        kill -TERM -"$TRAINPID" 2>/dev/null || kill -TERM "$TRAINPID" 2>/dev/null
        local waited=0
        while kill -0 "$TRAINPID" 2>/dev/null && [ "$waited" -lt 10 ]; do
            sleep 1
            waited=$(( waited + 1 ))
        done
        kill -KILL -"$TRAINPID" 2>/dev/null || kill -KILL "$TRAINPID" 2>/dev/null
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

: >"$LOG"

WATCHDOG=no
if command -v setsid >/dev/null 2>&1 && command -v pgrep >/dev/null 2>&1; then
    setsid bash "$SCRIPT" >"$LOG" 2>&1 &
    TRAINPID=$!
    WATCHDOG=yes
else
    log "setsid/pgrep unavailable; running without a stall watchdog"
    bash "$SCRIPT" >"$LOG" 2>&1 &
    TRAINPID=$!
fi

tail -f "$LOG" &
TAILPID=$!

START=$(date +%s)
STUCK=0
while kill -0 "$TRAINPID" 2>/dev/null; do
    BUSY=0
    for child in $(pgrep -P "$TRAINPID" 2>/dev/null); do
        [ "$(cat /proc/"$child"/comm 2>/dev/null)" = tee ] || BUSY=$(( BUSY + 1 ))
    done
    if [ "$BUSY" = 0 ]; then
        STUCK=$(( STUCK + 1 ))
    else
        STUCK=0
    fi
    [ "$STUCK" -ge "$STUCK_LIMIT" ] && break
    sleep "$POLL"
done

STALLED=no
if [ "$WATCHDOG" = yes ] && kill -0 "$TRAINPID" 2>/dev/null; then
    STALLED=yes
    log "trainer exited but $SCRIPT is stuck on tee; ~$(( STUCK * POLL ))s with no child but tee"
fi

if [ "$STALLED" = yes ]; then
    # cleanup() handles the group once we fall out of here.
    RC=1
else
    wait "$TRAINPID"
    RC=$?
fi

ELAPSED=$(( $(date +%s) - START ))

# Log lines look like {'loss': '1.225', ...} -- the number carries quotes. A regex
# without the optional quote silently matches nothing, which is how a previous run
# reported "no loss logged" after training had succeeded.
LAST_LOSS=$(grep -oE "'loss': '?[0-9.]+" "$LOG" 2>/dev/null | tail -1 | tr -d "'")

printf 'RC=%s\nELAPSED=%s\nWATCHDOG=%s\nSTUCK=%s\nLAST_LOSS=%s\n' \
    "$RC" "$ELAPSED" "$WATCHDOG" "$STALLED" "${LAST_LOSS:-}" >>"$OUT"

if [ "$ELAPSED" -gt 0 ]; then
    log "train rc=${RC} in ${ELAPSED}s (watchdog=${WATCHDOG} stalled=${STALLED})"
else
    log "train rc=${RC} (elapsed unmeasurable)"
fi

exit "$RC"
