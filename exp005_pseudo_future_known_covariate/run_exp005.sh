#!/usr/bin/env bash
# ===========================================================================
# FiNAR Experiment 5 — does a PSEUDO known-future covariate help a one-pass model?
# ===========================================================================
# Models trained WITHOUT iterative refinement gain a lot from covariates whose
# future is genuinely observed — fev-bench's future-known subset is where
# Chronos-2 reports its largest margin. This asks whether they gain anything
# when that future is not observed but PREDICTED, by the model itself:
#
#   step 1   multivariate forecast of every variate               (one pass)
#   step 2   variate 0 again, with variates 1..n-1 supplied as
#            past_covariates + future_covariates, the future filled from step 1
#
# Only variate 0 — the designated target — is scored, in BOTH steps, so the two
# numbers answer one question about one series.
#
# ---------------------------------------------------------------------------
# WHY THE TASK HAS TO BE REBUILT
# ---------------------------------------------------------------------------
# Neither evaluation set has a covariate slot to fill. fev-bench's
# `multivariate` subset is DEFINED as ">1 target column and no dynamic
# covariates", and GIFT-Eval has no covariates at all — its own adapter says
# so. Step 2 therefore restructures the task, moving variates 1..n-1 out of
# `target` and into the covariate dicts. That is also why the scoring is done
# here rather than by an adapter: step 1 scores n targets and step 2 scores
# one, so an adapter's aggregate would compare two different populations.
#
# STEP 2 MUST DIFFER FROM STEP 1. Chronos-2 and EO v4 resolve
# `supports_covariates` through the same function as `supports_multivariate`,
# so both steps can reach the model by one path and step 2 can silently return
# step 1 — which reads as "no gain" but is not one. Every run counts the items
# whose two forecasts are identical and REFUSES a run where all of them are.
#
# ---------------------------------------------------------------------------
# USAGE — ON THE REMOTE A100, OVER SSH (not as a submitted job)
# ---------------------------------------------------------------------------
#   ssh <a100-host>
#   cd /group-volume/workspace/mun-hak.lee/experiments/finar_001/finar/exp005_pseudo_future_known_covariate
#
#   # every model, both benchmarks, then table + figure
#   bash run_exp005.sh --repo /group-volume/.../tsm-trainer_001/tsm-trainer
#
#   # one model
#   bash run_exp005.sh --ckpt amazon/chronos-2
#   bash run_exp005.sh --ckpt google/timesfm-3.0-pytorch
#   bash run_exp005.sh --ckpt NX-AI/TiRex-2
#   bash run_exp005.sh --ckpt /path/to/eo-v4/best_checkpoints --depth 1
#
#   # a quick smoke, and tables only
#   bash run_exp005.sh --ckpt amazon/chronos-2 --max-tasks 2 --max-items 16
#   bash run_exp005.sh --stage table
#
# Long runs: detach, since an ssh drop kills the job.
#   nohup bash run_exp005.sh > exp005.log 2>&1 &
#   tail -f exp005.log
#
# Re-running is safe: a model whose csv exists is skipped.
#
# ---------------------------------------------------------------------------
# OPTIONS
# ---------------------------------------------------------------------------
#   --ckpt PATH|ID      one model. Repeatable. Default: all four below.
#                       amazon/chronos-2, google/timesfm-3.0-pytorch,
#                       NX-AI/TiRex-2, and a local EO v4 checkpoint.
#   --repo PATH         tsm-trainer checkout to IMPORT from. Read-only.
#   --stage eval|table|all        default: all
#   --benchmarks "fev gift"       default: both
#   --depth N           EO v4 only: --coe-eval-depth. Default 1, which makes a
#                       K>=2 checkpoint behave as the one-pass model this
#                       experiment is about. Ignored by the other three.
#   --batch-size N      default 32
#   --max-tasks N       cap tasks per benchmark (smoke runs)
#   --max-items N       cap items per task
#   --max-windows N     fev only, default 8. Its multivariate tasks carry very
#                       few SERIES — ETT is one per window — so a single window
#                       gives a two-item task and no statistic worth reading.
#   --task-subset "i n" score only task slice i of n
#   --out PATH          results dir, default ./results
#   --fev-data / --gift-data PATH    benchmark roots (default: local cache)
#   --dry-run           print the commands without running them
#
# ---------------------------------------------------------------------------
# WHAT COMES OUT
# ---------------------------------------------------------------------------
#   <out>/<model>.csv               per task: step1/step2 MASE, WQL, win rate
#   <out>/<model>.meta.json         supports_multivariate, identical-item count
#   <out>/exp005_table.csv          pooled per model and benchmark
#   <out>/exp005_improvement.png    the same as grouped bars
#
# ---------------------------------------------------------------------------
# A NOTE ON amazon/chronos-2
# ---------------------------------------------------------------------------
# That id routes to tsm-trainer's OFFICIAL-Chronos loader, which needs the
# `chronos-forecasting` package in the interpreter being used. Where it is
# absent, `autogluon/chronos-2` reaches the same model through
# Chronos2Forecaster and is what the local verification used.
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/group-volume/workspace/mun-hak.lee/experiments/tsm-trainer_001/tsm-trainer}"
DEFAULT_CKPTS=(
    "amazon/chronos-2"
    "google/timesfm-3.0-pytorch"
    "NX-AI/TiRex-2"
    "/group-volume/workspace/mun-hak.lee/experiments/tsm-trainer_001/tsm-trainer/outputs/eo-v4-120M-toto_2080ti/best_checkpoints"
)
CKPTS=(); STAGE="all"; DEPTH=1; BATCH=32; OUT=""
BENCHMARKS=""; MAX_TASKS=""; MAX_ITEMS=""; MAX_WINDOWS=""
TASK_SUBSET=""; FEV_DATA=""; GIFT_DATA=""; DRY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)        CKPTS+=("$2");   shift 2 ;;
        --repo)        REPO="$2";       shift 2 ;;
        --stage)       STAGE="$2";      shift 2 ;;
        --depth)       DEPTH="$2";      shift 2 ;;
        --batch-size)  BATCH="$2";      shift 2 ;;
        --out)         OUT="$2";        shift 2 ;;
        --benchmarks)  BENCHMARKS="$2"; shift 2 ;;
        --max-tasks)   MAX_TASKS="$2";  shift 2 ;;
        --max-items)   MAX_ITEMS="$2";  shift 2 ;;
        --max-windows) MAX_WINDOWS="$2"; shift 2 ;;
        --task-subset) TASK_SUBSET="$2"; shift 2 ;;
        --fev-data)    FEV_DATA="$2";   shift 2 ;;
        --gift-data)   GIFT_DATA="$2";  shift 2 ;;
        --dry-run)     DRY="echo [dry]"; shift ;;
        # To the end of the header, not a hard-coded line number: the header IS
        # the documentation and a fixed range truncates it as it grows.
        -h|--help)     sed -n '2,${/^set -/q;p;}' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -d "${REPO}" ]] || { echo "--repo ${REPO} is not a directory" >&2; exit 2; }
case "${STAGE}" in eval|table|all) ;; *)
    echo "--stage must be eval, table or all (got ${STAGE})" >&2; exit 2 ;;
esac
[[ ${#CKPTS[@]} -gt 0 ]] || CKPTS=("${DEFAULT_CKPTS[@]}")
PY="${PYTHON:-${REPO}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY="$(command -v python3)"
[[ -x "${PY}" ]] || { echo "no python found (tried ${REPO}/.venv/bin/python)" >&2; exit 1; }

[[ -n "${OUT}" ]] || OUT="${HERE}/results"
note() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

if [[ "${STAGE}" == "eval" || "${STAGE}" == "all" ]]; then
    for CKPT in "${CKPTS[@]}"; do
        TAG="$(basename "${CKPT}")"
        if [[ -f "${OUT}/${TAG}.csv" ]]; then
            note "SKIP ${CKPT} — ${OUT}/${TAG}.csv exists"
            continue
        fi
        note "=== ${CKPT}"
        # One model per process: a failure on one must not cost the others, and
        # each holds a whole checkpoint in VRAM.
        ${DRY} "${PY}" "${HERE}/run_eval.py" \
            --model-path "${CKPT}" --repo "${REPO}" --out "${OUT}" \
            --coe-eval-depth "${DEPTH}" --batch-size "${BATCH}" \
            ${BENCHMARKS:+--benchmarks ${BENCHMARKS}} \
            ${MAX_TASKS:+--max-tasks ${MAX_TASKS}} \
            ${MAX_ITEMS:+--max-items-per-task ${MAX_ITEMS}} \
            ${MAX_WINDOWS:+--max-windows ${MAX_WINDOWS}} \
            ${FEV_DATA:+--fev-data "${FEV_DATA}"} \
            ${GIFT_DATA:+--gift-data "${GIFT_DATA}"} \
            ${TASK_SUBSET:+--task-subset-index ${TASK_SUBSET% *} \
                           --num-task-subsets ${TASK_SUBSET#* }} \
            || note "FAILED: ${CKPT} (continuing with the rest)"
    done
fi

if [[ "${STAGE}" == "table" || "${STAGE}" == "all" ]]; then
    note "=== table + figure -> ${OUT}"
    ${DRY} "${PY}" "${HERE}/build_table.py" "${OUT}"
fi

note "done. results under ${OUT}"
