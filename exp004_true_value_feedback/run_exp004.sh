#!/usr/bin/env bash
# ===========================================================================
# FiNAR Experiment 4 — why does forecast-space iterative refining help?
# ===========================================================================
# EO v4's second pass is handed the first pass's forecast and predicts a
# residual against it. This replaces what it is handed:
#
#     intermediate_pred_new = (1 - alpha) * intermediate_pred + alpha * target
#
# and sweeps alpha, so the trend from "the model's own forecast" (alpha = 0,
# the stock run) to "the ground truth" (alpha = 1) can be read off. What the
# second pass does with a perfect input is the diagnostic: if it still cannot
# reach the truth, the refinement is limited by something other than the
# quality of its input.
#
# ---------------------------------------------------------------------------
# READ THIS BEFORE QUOTING ANY NUMBER
# ---------------------------------------------------------------------------
# EVERY alpha > 0 RESULT IS LABEL LEAKAGE BY CONSTRUCTION. fev's own API says
# of the data this feeds the model: "This data should never be provided to the
# model!". These rows are a diagnostic upper bound on what a second pass can do
# with a better input. They are NOT benchmark scores and must never be reported
# as model performance. Only the alpha = 0 row is comparable to anything else.
#
# ---------------------------------------------------------------------------
# USAGE — ON THE REMOTE A100, OVER SSH (not as a submitted job)
# ---------------------------------------------------------------------------
#   ssh <a100-host>
#   cd /group-volume/workspace/mun-hak.lee/experiments/finar_001/finar/exp004_true_value_feedback
#
#   # everything: sweep alpha on both benchmarks, then table + figure
#   bash run_exp004.sh --ckpt /path/to/eo-v4-K2/best_checkpoints \
#                      --repo /group-volume/.../tsm-trainer_001/tsm-trainer
#
#   bash run_exp004.sh --ckpt <ckpt> --alpha "0.0 0.25 0.5 0.75 1.0"
#   bash run_exp004.sh --ckpt <ckpt> --benchmarks fev
#   bash run_exp004.sh --ckpt <ckpt> --task-subset "0 4"   # slice 0 of 4
#   bash run_exp004.sh --stage table --out <results dir>   # table only, no ckpt
#
#   # error accumulation: score every depth 1..16 and read the trajectory
#   bash run_exp004.sh --ckpt <ckpt> --alpha "0.0 0.5 1.0" --depth 16
#
# Long runs: detach, since an ssh drop kills the job.
#   nohup bash run_exp004.sh --ckpt <ckpt> > exp004.log 2>&1 &
#   tail -f exp004.log
#
# Re-running is safe: a (benchmark, alpha) whose results.csv exists is skipped.
# A re-run with a different checkpoint, depth, fev subset or task slice into the
# same --out is REFUSED rather than silently mixing arms — manifest.json records
# what the directory was built from.
#
# ---------------------------------------------------------------------------
# OPTIONS
# ---------------------------------------------------------------------------
#   --ckpt PATH         EO v4 checkpoint to score. Required.
#   --repo PATH         tsm-trainer checkout to IMPORT from. Read-only.
#   --threads N         CPU threads this run may use. Default 8.
#                       Use 8, 16 or 32; the pools are capped BEFORE
#                       python starts, which is the only time it works
#                       for OpenMP/BLAS.
#   --stage eval|table|all        default: all
#   --alpha "0.0 0.5 1.0"         blend weights to sweep (default: those three)
#   --benchmarks "fev gift"       default: both
#   --fev-subset NAME   all | univariate | multivariate | covariate.
#                       Default `multivariate`: every variate is then a target,
#                       so "the truth is fed back for every variate" has no
#                       covariate exception.
#   --depth N           --coe-eval-depth, default 2 (must be >= 2). Every depth
#                       1..N is scored and reported, so this is also the knob
#                       for the error-accumulation sweep: raise it and read the
#                       MASE trajectory across iterations. Verified working at
#                       16 on a K=2 checkpoint (GPU peak 0.61 GB, vs 0.57 GB at
#                       4) — a depth above the trained one is extrapolation,
#                       recorded and not refused.
#   --batch-size N      default 64
#   --out PATH          results dir, default ./results/<ckpt basename>
#   --task-subset "i n" score only task slice i of n. GIFT-Eval does not fit in
#                       23 GB in one pass on a workstation; the slice is
#                       recorded in manifest.json, since two runs are only
#                       comparable if they scored the same tasks.
#   --fev-data PATH / --gift-data PATH   benchmark roots (default: local cache)
#   --dry-run           print the commands without running them
#
# ---------------------------------------------------------------------------
# WHAT COMES OUT
# ---------------------------------------------------------------------------
#   <out>/manifest.json                  checkpoint, depth, subset, slice
#   <out>/<bench>/alpha<a>/results.csv   per-task scores, every depth
#   <out>/<bench>/alpha<a>/truth.json    how often the truth reached the model
#   <out>/exp004_table.csv               iter1 vs iter2 per alpha
#   <out>/exp004_improvement.png         improvement against alpha
#
# ---------------------------------------------------------------------------
# CHECKPOINT REQUIREMENTS
# ---------------------------------------------------------------------------
# `coe_residual: True` — pass 2 predicts a residual against an accumulator
# seeded from what it was handed, and the whole design rests on that
# accumulator being re-derived from the BLENDED median. Without it there is no
# residual to keep consistent and run_eval.py refuses the checkpoint.
# `coe_bottleneck: True` — without it the passes chain in hidden space and
# never write a forecast back, so there is no intermediate prediction to
# replace. `coe_train_depth_max >= 2` — otherwise there is no second pass.
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/group-volume/workspace/mun-hak.lee/experiments/tsm-trainer_001/tsm-trainer}"
THREADS="${THREADS:-8}"
CKPT=""; STAGE="all"; DEPTH=2; BATCH=64; OUT=""
ALPHA="0.0 0.5 1.0"; BENCHMARKS=""; FEV_SUBSET=""
FEV_DATA=""; GIFT_DATA=""; TASK_SUBSET=""; DRY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)        CKPT="$2";       shift 2 ;;
        --repo)        REPO="$2";       shift 2 ;;
        --threads)     THREADS="$2";   shift 2 ;;
        --stage)       STAGE="$2";      shift 2 ;;
        --alpha)       ALPHA="$2";      shift 2 ;;
        --depth)       DEPTH="$2";      shift 2 ;;
        --batch-size)  BATCH="$2";      shift 2 ;;
        --out)         OUT="$2";        shift 2 ;;
        --benchmarks)  BENCHMARKS="$2"; shift 2 ;;
        --fev-subset)  FEV_SUBSET="$2"; shift 2 ;;
        --fev-data)    FEV_DATA="$2";   shift 2 ;;
        --gift-data)   GIFT_DATA="$2";  shift 2 ;;
        --task-subset) TASK_SUBSET="$2"; shift 2 ;;
        --dry-run)     DRY="echo [dry]"; shift ;;
        # To the end of the header, not to a hard-coded line number: the header
        # IS the documentation and a fixed range truncates it as it grows.
        -h|--help)     sed -n '2,${/^set -/q;p;}' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -d "${REPO}" ]] || { echo "--repo ${REPO} is not a directory" >&2; exit 2; }
case "${STAGE}" in eval|table|all) ;; *)
    echo "--stage must be eval, table or all (got ${STAGE})" >&2; exit 2 ;;
esac
# Required only where it is used. --stage table reads a finished --out and
# never loads a model, so demanding a checkpoint there forced a dummy path.
if [[ "${STAGE}" != "table" ]]; then
    [[ -n "${CKPT}" ]] || { echo "--ckpt is required for --stage ${STAGE}" >&2; exit 2; }
else
    [[ -n "${OUT}" || -n "${CKPT}" ]] || {
        echo "--stage table needs --out (the results dir to table) or --ckpt" >&2
        exit 2; }
fi
PY="${PYTHON:-${REPO}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY="$(command -v python3)"
[[ -x "${PY}" ]] || { echo "no python found (tried ${REPO}/.venv/bin/python)" >&2; exit 1; }

[[ -n "${OUT}" ]] || OUT="${HERE}/results/$(basename "${CKPT}")"
# CPU CAP — EXPORTED BEFORE PYTHON STARTS, which is the only time it works for
# OpenMP/BLAS: those size their pools when the library is first loaded, so a
# value set after `import torch` is ignored. finar_cpu.py handles what CAN be
# set at runtime (torch, pyarrow, fev). multiprocessing.cpu_count() also ignores
# cgroup quotas, so on a node reporting 255 cores while allocating ~29 the
# uncapped pools size to 255 and the job is killed rather than merely slow.
[[ "${THREADS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "--threads must be a positive integer (got '${THREADS}')" >&2; exit 2; }
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_MAX_THREADS="${THREADS}"
export TOKIO_WORKER_THREADS="${THREADS}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${THREADS}"
export TOKENIZERS_PARALLELISM=false

note() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

if [[ "${STAGE}" == "eval" || "${STAGE}" == "all" ]]; then
    note "=== scoring ${CKPT} over alpha ${ALPHA} -> ${OUT}"
    ${DRY} "${PY}" "${HERE}/run_eval.py" \
        --num-workers "${THREADS}" \
        --model-path "${CKPT}" --repo "${REPO}" --out "${OUT}" \
        --alpha ${ALPHA} --coe-eval-depth "${DEPTH}" --batch-size "${BATCH}" \
        ${BENCHMARKS:+--benchmarks ${BENCHMARKS}} \
        ${FEV_SUBSET:+--fev-subset "${FEV_SUBSET}"} \
        ${FEV_DATA:+--fev-data "${FEV_DATA}"} \
        ${GIFT_DATA:+--gift-data "${GIFT_DATA}"} \
        ${TASK_SUBSET:+--task-subset-index ${TASK_SUBSET% *} \
                       --num-task-subsets ${TASK_SUBSET#* }}
fi

if [[ "${STAGE}" == "table" || "${STAGE}" == "all" ]]; then
    note "=== table + figure -> ${OUT}"
    # No --iters: build_table.py reads the depth the run was scored at from
    # <out>/manifest.json. Passing ${DEPTH} here tabled iterations 1..2 of a
    # depth-16 run whenever --stage table was invoked without repeating --depth.
    ${DRY} "${PY}" "${HERE}/build_table.py" "${OUT}"
fi

note "done. results under ${OUT}"
