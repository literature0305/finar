#!/usr/bin/env bash
# ===========================================================================
# FiNAR Experiment 1 — build the 3x3x3 grid, score it, and table the result.
# ===========================================================================
# Tests the FiNAR hypothesis that iterative refinement supplies an inductive
# bias for dependency INSIDE the forecast target, over three axes:
#
#     dependency along time      phi   in {low 0.3, mid 0.6, high 0.9}
#     cross-variate, by re-pairing variates across items
#                                shuf  in {none, half, all}
#     cross-variate, by shrinking the group that shares a latent bank
#                                group in {full 4, half 2, no 1}
#
# 27 cells, ONE length (2048). The horizon is an EVALUATION-TIME SLICE, not a
# corpus: H in {16, 64, 256, 1024} are views of the same data, all scored on
# the same 512-step context window. The previous build generated a separate
# corpus per horizon, so a trend down the H axis compared different draws.
#
# Two tables come out, one per cross-variate axis; they must agree where both
# reach zero correlation (`all` shuffle and `no` group), which is the design's
# own consistency check.
#
# ---------------------------------------------------------------------------
# USAGE — ON THE REMOTE A100, OVER SSH (not as a submitted job)
# ---------------------------------------------------------------------------
#   ssh <a100-host>
#   cd /group-volume/workspace/mun-hak.lee/experiments/finar_001/finar/exp001_synthetic_data
#
#   # 1. build the corpora once (~CPU only, no GPU needed)
#   bash run_exp001.sh --stage data
#
#   # 2. score the EO v4 checkpoint (needs a K>=2 checkpoint; see DEPTH below)
#   bash run_exp001.sh --stage eval --ckpt /path/to/eo-v4-K2/best_checkpoints
#
#   # 3. Chronos-2 baseline — no iteration axis, so it needs --allow-shallow
#   bash run_exp001.sh --stage eval --ckpt autogluon/chronos-2 --shallow
#
#   # 4. tables
#   bash run_exp001.sh --stage table
#
#   # everything in one go
#   bash run_exp001.sh --stage all --ckpt /path/to/eo-v4-K2/best_checkpoints
#
# Long runs: prefix with nohup and detach, since ssh drops kill the job.
#   nohup bash run_exp001.sh --stage all --ckpt <ckpt> > exp001.log 2>&1 &
#   tail -f exp001.log
#
# ---------------------------------------------------------------------------
# OPTIONS
# ---------------------------------------------------------------------------
#   --stage data|eval|table|all   which part to run (default: all)
#   --ckpt PATH                   model to score. Required for eval.
#   --depth N                     --coe-eval-depth (default: 4). Reports every
#                                 depth 1..N. Values above the checkpoint's
#                                 coe_train_depth_max are the EXTRAPOLATION arm
#                                 and are warned about, not refused.
#   --shallow                     score a model with no iteration axis
#                                 (Chronos-2, or a K=1 EO) as a baseline
#   --horizons "16 64"            subset of the horizon sweep
#   --context N                   observed steps before every horizon
#                                 (default: 512). Held EQUAL across horizons.
#                                 512 is the shortest context at which all four
#                                 horizons are measurable at all — from 64 an
#                                 oracle that grid-searches the period scores
#                                 R^2 = -7 at H=256 and -944 at H=1024, i.e.
#                                 worse than predicting the mean.
#   --data-root PATH              corpora location
#                                 (default: /group-volume/ts-dataset/finar_exp001)
#   --out PATH                    results dir (default: ./results/<model name>)
#   --num-series N                items per cell (default: 2048)
#   --shards N                    sibling corpora per cell (default: 8). This
#                                 is the denominator of every dispersion
#                                 statistic: the evaluator emits one row per
#                                 dataset, so an unsharded cell gives n=1 and
#                                 an improvement with no spread beside it.
#   --force-data                  rebuild the corpora even if they exist.
#                                 Without it --stage all SKIPS generation when
#                                 <data-root>/metadata.json is present.
#   --repo PATH                   tsm-trainer checkout to IMPORT from
#   --dry-run                     print the commands without running them
#
# ---------------------------------------------------------------------------
# DEPTH — READ THIS BEFORE BLAMING THE TABLE
# ---------------------------------------------------------------------------
# The experiment needs a checkpoint trained with coe_train_depth_max >= 2.
# With K=1 there is no second pass to report, and the failure is SILENT in
# tsm-trainer: `_repeats_worth_asking` returns False the moment
# `report_depth() == 1`, BEFORE it reads TSM_FEV_COE_REPEATS, and the log line
# that would explain the absence is itself gated on having depths to report.
# `run_eval.py` therefore reads the checkpoint's config and refuses up front.
#
# Of the eo-v4 configs in tsm-trainer, only 25M (K=6), 35M (K=4), 94M (K=2) and
# toto-4m-smoke (K=3) qualify; 120M-toto, 60M-toto, base, 44M, 63M, 15M and the
# toto-* family are all K=1.
#
# ---------------------------------------------------------------------------
# WHY THIS DOES NOT CALL run_evaluation.sh
# ---------------------------------------------------------------------------
# `run_benchmark.py` resolves every benchmark yaml from a hard-coded
# `configs_dir = SCRIPT_DIR / "configs"`, so routing an external corpus through
# it would mean writing a yaml INSIDE tsm-trainer. `run_eval.py` calls
# `Evaluator.evaluate_benchmark(config_path=...)` directly instead, which takes
# an arbitrary path. tsm-trainer is imported, never modified.
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/group-volume/workspace/mun-hak.lee/experiments/tsm-trainer_001/tsm-trainer}"
DATA_ROOT="/group-volume/ts-dataset/finar_exp001"
STAGE="all"
CKPT=""
DEPTH=4
SHALLOW=""
HORIZONS=""
OUT=""
NUM_SERIES=2048
SHARDS=8
CONTEXT=512
FORCE_DATA=""
DRY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)      STAGE="$2";      shift 2 ;;
        --ckpt)       CKPT="$2";       shift 2 ;;
        --depth)      DEPTH="$2";      shift 2 ;;
        --shallow)    SHALLOW="--allow-shallow"; shift ;;
        --horizons)   HORIZONS="$2";   shift 2 ;;
        --context)    CONTEXT="$2";    shift 2 ;;
        --data-root)  DATA_ROOT="$2";  shift 2 ;;
        --out)        OUT="$2";        shift 2 ;;
        --num-series) NUM_SERIES="$2"; shift 2 ;;
        --shards)     SHARDS="$2";     shift 2 ;;
        --force-data) FORCE_DATA=1;    shift ;;
        --repo)       REPO="$2";       shift 2 ;;
        --dry-run)    DRY="echo [dry]"; shift ;;
        # To the end of the header block, not to a hard-coded line number:
        # the header IS the documentation, and a fixed range silently starts
        # truncating it the moment it grows.
        -h|--help)    sed -n '2,${/^set -/q;p;}' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

PY="${PYTHON:-${REPO}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY="$(command -v python3)"
[[ -x "${PY}" ]] || { echo "no python found (tried ${REPO}/.venv/bin/python)" >&2; exit 1; }

if [[ "${STAGE}" != "data" && "${STAGE}" != "table" && -z "${CKPT}" ]]; then
    echo "--ckpt is required for --stage ${STAGE}" >&2; exit 2
fi
# Results land under a directory named for the model, so the EO run and the
# Chronos-2 baseline can sit side by side and `build_table.py` can take both.
[[ -n "${OUT}" ]] || OUT="${HERE}/results/$(basename "${CKPT:-none}")"

note() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

if [[ "${STAGE}" == "data" || "${STAGE}" == "all" ]]; then
    # --stage all is the normal way to run this, and rebuilding 14 GB every
    # time would also RESEED the corpora, so a re-run would score different
    # data than the run before it. `--stage data --force-data` rebuilds.
    # `|| true` inside the substitution: this script runs under `set -o
    # pipefail`, so a bare `ls` that matches nothing exits 2 and takes the
    # whole run down before it can report that there is nothing to skip.
    N_CELLS=$({ ls -d ${DATA_ROOT}/phi*_sh*_gp*_s* 2>/dev/null || true; } | wc -l)
    N_STALE=$({ ls -d ${DATA_ROOT}/H*_phi*_rho* 2>/dev/null || true; } | wc -l)
    # Counts the CORPORA, not metadata.json: the previous grid wrote its own
    # metadata.json under this same root with a completely different cell
    # naming, so keying the skip on that file would skip a build that has not
    # happened and then score 0 cells.
    if [[ "${N_STALE}" -gt 0 ]]; then
        note "WARNING: ${DATA_ROOT} holds ${N_STALE} corpora from the previous"
        note "         H*_phi*_rho* grid. They are not read by this pipeline"
        note "         and are not comparable to it; remove them when you are"
        note "         done with the old results."
    fi
    if [[ "${N_CELLS}" -gt 0 && -z "${FORCE_DATA}" ]]; then
        note "SKIP data: ${N_CELLS} corpora already under ${DATA_ROOT}."
        note "     --force-data to rebuild (which RESEEDS: a rebuilt corpus is"
        note "     not the one the existing results were scored on)."
    else
    note "=== building the 27-cell grid -> ${DATA_ROOT}"
    ${DRY} "${PY}" "${HERE}/build_dataset.py" \
        --out-root "${DATA_ROOT}" --num-series "${NUM_SERIES}" \
        --shards "${SHARDS}"
    fi
fi

if [[ "${STAGE}" == "eval" || "${STAGE}" == "all" ]]; then
    note "=== scoring ${CKPT} at depths 1..${DEPTH} -> ${OUT}"
    ${DRY} "${PY}" "${HERE}/run_eval.py" \
        --model-path "${CKPT}" --data-root "${DATA_ROOT}" \
        --out "${OUT}" --repo "${REPO}" --coe-eval-depth "${DEPTH}" \
        --context "${CONTEXT}" ${SHALLOW} ${HORIZONS:+--horizons ${HORIZONS}}
fi

if [[ "${STAGE}" == "table" || "${STAGE}" == "all" ]]; then
    note "=== tables"
    ${DRY} "${PY}" "${HERE}/build_table.py" "${HERE}/results" \
        --data-root "${DATA_ROOT}"
fi

note "done. results under ${HERE}/results"
