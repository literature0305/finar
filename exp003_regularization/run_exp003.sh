#!/usr/bin/env bash
# ===========================================================================
# FiNAR Experiment 3 — is the second pass modelling dependency, or regularising?
# ===========================================================================
# Builds a 5,000-series subset of a TRAINING mixture (not a benchmark) and
# scores a model on it at three horizons, at every recursion depth:
#
#     H =  48   3 patches
#     H = 160  10 patches
#     H = 320  20 patches       (input_patch_size 16)
#
# The subset is drawn with tsm-trainer's OWN sampling rule — a dataset is
# picked in proportion to data_points x custom_weight, then a row uniformly
# inside it — and restricted to series of at least 500 steps.
#
# WHY A HORIZON SWEEP DISTINGUISHES THE TWO EXPLANATIONS
# A regulariser should help roughly uniformly across horizons. Dependency
# modelling should help MORE as the horizon lengthens, because a single pass
# has more joint structure to get wrong at once. The shape of the iter1->iter2
# gain over H is the evidence.
#
# ---------------------------------------------------------------------------
# USAGE — ON THE REMOTE A100, OVER SSH (not as a submitted job)
# ---------------------------------------------------------------------------
#   ssh <a100-host>
#   cd /group-volume/workspace/mun-hak.lee/experiments/finar_001/finar/exp003_regularization
#
#   # 1. build the corpus once (CPU only; ~2 min of scanning per mixture)
#   bash run_exp003.sh --stage data \
#       --train-config <tsm-trainer>/scripts/forecasting/training/configs/eo-v4-120M-toto_2080ti.yaml \
#       --repo <tsm-trainer>
#
#   # 2. score a K>=2 checkpoint
#   bash run_exp003.sh --stage eval --ckpt <ckpt>/best_checkpoints --repo <tsm-trainer>
#
#   # 3. a baseline with no iteration axis
#   bash run_exp003.sh --stage eval --ckpt autogluon/chronos-2 --shallow --repo <tsm-trainer>
#
#   # everything
#   bash run_exp003.sh --ckpt <ckpt>/best_checkpoints --repo <tsm-trainer> \
#       --train-config <training yaml>
#
# Long runs: detach, since an ssh drop kills the job.
#   nohup bash run_exp003.sh --ckpt <ckpt> --repo <repo> > exp003.log 2>&1 &
#
# ---------------------------------------------------------------------------
# OPTIONS
# ---------------------------------------------------------------------------
#   --ckpt PATH           model to score. Required for eval.
#   --repo PATH           tsm-trainer checkout to IMPORT from. Read-only.
#   --num-workers N     CPU cap for this run. Default: read from the
#                       scheduler affinity mask and the cgroup quota,
#                       so a quota'd node configures itself. A value
#                       above the allocation is clamped, not obeyed.
#   --train-config PATH   training yaml whose training_data defines the mixture.
#                         Required for --stage data.
#   --stage data|eval|all default: all
#   --depth N             --coe-eval-depth, default 2
#   --shallow             score a model with no iteration axis
#   --n N                 series to draw, default 5000
#   --min-length N        shortest series to keep, default 500
#   --out-root PATH       where the corpus goes, default /group-volume/ts-dataset
#   --out PATH            results dir, default ./results/<ckpt basename>
#   --force-data          rebuild the corpus even if it exists
#   --dry-run             print the commands without running them
#
# ---------------------------------------------------------------------------
# TWO THINGS TO KNOW BEFORE READING THE NUMBERS
# ---------------------------------------------------------------------------
# 1. The mixture is what the yaml says it is, and for the EO v4 configs that is
#    dominated by SYNTHETIC corpora — tsmixup and the 100-variate multivariatizer
#    together carry ~93% of the sampling weight, so the drawn subset is mostly
#    synthetic. That is faithful to training, and it is also a limit on what the
#    result generalises to. build_subset.py prints the realised share per
#    dataset, and the manifest records all of it.
# 2. At H=320 a 500-step series leaves 180 steps of context. The long horizon is
#    the hardest of the three by construction; that spread is the sweep's point.
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/group-volume/workspace/mun-hak.lee/experiments/tsm-trainer_001/tsm-trainer}"
THREADS="${THREADS:-}"   # empty = derive from the allocation
CKPT=""; TRAIN_CFG=""; STAGE="all"; DEPTH=2; SHALLOW=""
N=5000; MIN_LEN=500; OUT_ROOT="/group-volume/ts-dataset"; OUT=""
FORCE_DATA=""; DRY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)         CKPT="$2";      shift 2 ;;
        --repo)         REPO="$2";      shift 2 ;;
        --num-workers) THREADS="$2";   shift 2 ;;
        --train-config) TRAIN_CFG="$2"; shift 2 ;;
        --stage)        STAGE="$2";     shift 2 ;;
        --depth)        DEPTH="$2";     shift 2 ;;
        --shallow)      SHALLOW="--allow-shallow"; shift ;;
        --n)            N="$2";         shift 2 ;;
        --min-length)   MIN_LEN="$2";   shift 2 ;;
        --out-root)     OUT_ROOT="$2";  shift 2 ;;
        --out)          OUT="$2";       shift 2 ;;
        --force-data)   FORCE_DATA=1;   shift ;;
        --dry-run)      DRY="echo [dry]"; shift ;;
        -h|--help)      sed -n '2,75p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -d "${REPO}" ]] || { echo "--repo ${REPO} is not a directory" >&2; exit 2; }
PY="${PYTHON:-${REPO}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY="$(command -v python3)"
[[ -x "${PY}" ]] || { echo "no python found (tried ${REPO}/.venv/bin/python)" >&2; exit 1; }

if [[ "${STAGE}" != "data" && -z "${CKPT}" ]]; then
    echo "--ckpt is required for --stage ${STAGE}" >&2; exit 2
fi
[[ -n "${OUT}" ]] || OUT="${HERE}/results/$(basename "${CKPT:-none}")"
# CPU cap: one definition, sourced. See finar_cpu.sh / finar_cpu.py.
. "$(dirname "${HERE}")/finar_cpu.sh"

note() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

if [[ "${STAGE}" == "data" || "${STAGE}" == "all" ]]; then
    if [[ -d "${OUT_ROOT}/train_subset_5k" && -z "${FORCE_DATA}" ]]; then
        note "SKIP data: ${OUT_ROOT}/train_subset_5k exists. --force-data to rebuild."
    elif [[ -z "${TRAIN_CFG}" ]]; then
        echo "--train-config is required to build the corpus" >&2; exit 2
    else
        note "=== drawing ${N} series (>= ${MIN_LEN} steps) -> ${OUT_ROOT}/train_subset_5k"
        ${DRY} "${PY}" "${HERE}/build_subset.py" \
            --config "${TRAIN_CFG}" --repo "${REPO}" --out-root "${OUT_ROOT}" \
            --n "${N}" --min-length "${MIN_LEN}"
    fi
fi

if [[ "${STAGE}" == "eval" || "${STAGE}" == "all" ]]; then
    note "=== scoring ${CKPT} at depths 1..${DEPTH} -> ${OUT}"
    ${DRY} "${PY}" "${HERE}/run_eval.py" \
        ${THREADS:+--num-workers "${THREADS}"} \
        --model-path "${CKPT}" --repo "${REPO}" --out "${OUT}" \
        --coe-eval-depth "${DEPTH}" ${SHALLOW}
fi

note "done. results under ${OUT}"
