#!/usr/bin/env bash
# ===========================================================================
# exp007 — score existing checkpoints, and table the comparison
# ===========================================================================
# `train.sh` already evaluates every run it finishes, so this exists for the
# two things that come afterwards: scoring a checkpoint again at DIFFERENT
# chain depths (test-time scaling costs nothing to re-measure and needs no
# retraining), and collecting every run into one table and figure.
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   # re-score everything under ./runs at depths 1..4, then table it
#   bash eval.sh --depths "1 2 3 4"
#
#   # one checkpoint
#   bash eval.sh --run-dir runs/ETTh1_96_96_coe3-res-bn --depths "1 2 3 4"
#
#   # table only — no model is loaded
#   bash eval.sh --stage table
#
# A baseline checkpoint has no forecast slot to refine, so --depths above 1 is
# REFUSED for one rather than silently scoring depth 1 four times.
#
# ---------------------------------------------------------------------------
# OPTIONS
# ---------------------------------------------------------------------------
#   --run-dir PATH      one run directory. Repeatable. Default: every run
#                       directory under --out
#   --out PATH          results root (default ./runs)
#   --data-root PATH    dataset root (default /group-volume/ts-dataset/ltsf)
#   --depths "1 2 3 4"  chain depths to score, in one forward per batch.
#                       Default: the checkpoint's own coe_eval_depth
#   --batch-size N      eval batch size (default 32). The official loader uses
#                       1; MSE/MAE are means over every window either way
#   --num-workers N     CPU cap. Default: read from the scheduler affinity mask
#                       and the cgroup quota. Above the allocation it is
#                       clamped, not obeyed.
#   --loader-workers N  DataLoader worker processes (default 0); scoring is a
#                       single pass over an in-memory array
#   --device cuda|cpu   default: cuda when available
#   --stage eval|table|all      default: all
#   --dry-run           print the commands without running them
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/group-volume/ts-dataset/ltsf}"
THREADS="${THREADS:-}"   # empty = derive from the allocation
RUN_DIRS=(); OUT=""; DEPTHS=""; BATCH="32"; LOADER_WORKERS=""
DEVICE=""; STAGE="all"; DRY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)        RUN_DIRS+=("$2");   shift 2 ;;
        --out)            OUT="$2";           shift 2 ;;
        --data-root)      DATA_ROOT="$2";     shift 2 ;;
        --depths)         DEPTHS="$2";        shift 2 ;;
        --batch-size)     BATCH="$2";         shift 2 ;;
        --num-workers)    THREADS="$2";       shift 2 ;;
        --loader-workers) LOADER_WORKERS="$2"; shift 2 ;;
        --device)         DEVICE="$2";        shift 2 ;;
        --stage)          STAGE="$2";         shift 2 ;;
        --dry-run)        DRY="echo [dry]";   shift ;;
        -h|--help)        sed -n '2,${/^set -/q;p;}' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "${STAGE}" in eval|table|all) ;; *)
    echo "--stage must be eval, table or all (got ${STAGE})" >&2; exit 2 ;;
esac

PY="${PYTHON:-}"
if [[ -z "${PY}" ]]; then
    for CAND in "${HERE}/.venv/bin/python" \
                "/group-volume/workspace/mun-hak.lee/experiments/tsm-trainer_001/tsm-trainer/.venv/bin/python" \
                "$(command -v python3 || true)"; do
        [[ -x "${CAND}" ]] && { PY="${CAND}"; break; }
    done
fi
[[ -x "${PY}" ]] || { echo "no python found; set PYTHON=..." >&2; exit 1; }

[[ -n "${OUT}" ]] || OUT="${HERE}/runs"
# CPU cap: one definition, sourced. See finar_cpu.sh / finar_cpu.py.
. "$(dirname "${HERE}")/finar_cpu.sh"

note() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

if [[ "${STAGE}" == "eval" || "${STAGE}" == "all" ]]; then
    if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
        # A run directory is one with a checkpoint in it; globbing the parent
        # would also pick up the table and figure written beside them.
        while IFS= read -r CKPT; do
            RUN_DIRS+=("$(dirname "${CKPT}")")
        done < <(find "${OUT}" -maxdepth 2 -name checkpoint.pt | sort)
    fi
    [[ ${#RUN_DIRS[@]} -gt 0 ]] || {
        note "no run directories under ${OUT} — nothing to score"; }
    for RUN in "${RUN_DIRS[@]:-}"; do
        [[ -n "${RUN}" ]] || continue
        note "=== $(basename "${RUN}")"
        ${DRY} "${PY}" "${HERE}/run_eval.py" \
            --run-dir "${RUN}" --data-root "${DATA_ROOT}" \
            --batch-size "${BATCH}" \
            ${THREADS:+--num-workers "${THREADS}"} \
            ${DEPTHS:+--depths ${DEPTHS}} \
            ${LOADER_WORKERS:+--loader-workers "${LOADER_WORKERS}"} \
            ${DEVICE:+--device "${DEVICE}"} \
            || note "FAILED: ${RUN} (continuing with the rest)"
    done
fi

if [[ "${STAGE}" == "table" || "${STAGE}" == "all" ]]; then
    note "=== table + figure -> ${OUT}"
    ${DRY} "${PY}" "${HERE}/build_table.py" "${OUT}"
fi

note "done. results under ${OUT}"
