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
#   bash run_exp005.sh --ckpt Datadog/Toto-2.0-313m
#   bash run_exp005.sh --ckpt /path/to/eo-v4/best_checkpoints --depth 1
#
#   # Toto-2.0 across sizes (see the note at the bottom for why it belongs here)
#   for S in 4m 22m 313m 1B 2.5B; do
#       bash run_exp005.sh --ckpt "Datadog/Toto-2.0-$S"
#   done
#
#   # oracle-vs-pseudo sweep: does a BETTER covariate forecast help?
#   bash run_exp005.sh --ckpt <ckpt> --alpha "0 0.25 0.5 0.75 1"
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
#   --ckpt PATH|ID      one model. Repeatable. Default: all five below.
#                       amazon/chronos-2, google/timesfm-3.0-pytorch,
#                       NX-AI/TiRex-2, Datadog/Toto-2.0-313m, and a local EO v4
#                       checkpoint. Any Datadog/Toto-2.0-{4m,22m,313m,1B,2.5B,
#                       2.5B-FT} is accepted — the loader dispatches on the
#                       "toto-2" substring, so the size is free to vary.
#   --repo PATH         tsm-trainer checkout to IMPORT from. Read-only.
#   --threads N         CPU threads this run may use. Default 8.
#                       Use 8, 16 or 32; the pools are capped BEFORE
#                       python starts, which is the only time it works
#                       for OpenMP/BLAS.
#   --stage eval|table|all        default: all
#   --benchmarks "fev gift"       default: both
#   --depth N           EO v4 only: --coe-eval-depth. Default 1, which makes a
#                       K>=2 checkpoint behave as the one-pass model this
#                       experiment is about. Ignored by the other models.
#   --alpha "0 0.25 0.5 0.75 1"
#                       How much of the covariates' TRUE future to blend into
#                       the model's own forecast of them:
#                           alpha*oracle + (1-alpha)*pseudo
#                       0 (the default) is the original pseudo scenario; 1 hands
#                       the model the real known future for its covariates. The
#                       SCORED TARGET is never oracle. Every alpha shares one
#                       step 1, so the baseline cannot drift between columns.
#                       alpha > 0 is label leakage on the covariates — an upper
#                       bound on what a perfect covariate forecaster could buy,
#                       not a score.
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
#   <out>/<model>.meta.json         supports_multivariate, alpha grid,
#                                   identical/items per alpha
#   <out>/exp005_table.csv          pooled per model and benchmark
#   <out>/exp005_improvement.png    the same as grouped bars
#   <out>/exp005_alpha.png          MASE / improvement / win rate vs alpha
#
# ---------------------------------------------------------------------------
# A NOTE ON amazon/chronos-2
# ---------------------------------------------------------------------------
# That id routes to tsm-trainer's OFFICIAL-Chronos loader, which needs the
# `chronos-forecasting` package in the interpreter being used. Where it is
# absent, `autogluon/chronos-2` reaches the same model through
# Chronos2Forecaster and is what the local verification used.
#
# ---------------------------------------------------------------------------
# A NOTE ON Datadog/Toto-2.0
# ---------------------------------------------------------------------------
# Toto-2 belongs in this experiment because it takes a REAL known future rather
# than only extra past variates: Toto2Forecaster feeds known-dynamic covariates
# through the model's `known_dynamic` slot, spanning context+horizon, so their
# future half conditions the forecast (engine/forecaster.py, _forecast_tasks_
# batch). Verified directly — supplying a future changes the forecast, and
# REVERSING that future changes it again, so the model reads the values and not
# merely the slot.
#
# Needs `pip install 'toto-2 @ git+https://github.com/DataDog/toto.git#subdirect
# ory=toto2'` (--no-deps, or it clobbers torch). Any size works: the loader
# dispatches on the "toto-2" substring of the id.
#
# HORIZON LIMIT — READ THIS WITH EVERY TOTO-2 ROW. Toto2Forecaster reaches the
# model with only the FIRST ceil(H/32)-1 patches of the known future (32 is
# Toto's patch_size), so the last patch of the horizon never sees its future
# and, for H <= 32, none of it does. Measured by perturbing one future patch at
# a time: at H=96 patches 0 and 1 move the forecast and patch 2 does not; the
# threshold is H>=33 at every context length tried. 15 of exp005's 45 tasks
# have H <= 32, and a Toto-2 row on those is a guaranteed null — run_eval.py
# WARNs per task and the n_identical column carries it into the table. This is
# a tsm-trainer defect; it is reported, not patched from here.
#
# Its PUBLISHED fev-bench numbers use no covariates at all — the official
# adapter builds Toto2GluonTSModelConfig with no covariate dims — so this run
# is deliberately NOT the leaderboard protocol. That is the default here;
# --leaderboard-parity in run_benchmark.py is what restores it.
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/group-volume/workspace/mun-hak.lee/experiments/tsm-trainer_001/tsm-trainer}"
THREADS="${THREADS:-8}"
DEFAULT_CKPTS=(
    "amazon/chronos-2"
    "google/timesfm-3.0-pytorch"
    "NX-AI/TiRex-2"
    # One size by default; the header shows the loop over the other five.
    "Datadog/Toto-2.0-313m"
    "/group-volume/workspace/mun-hak.lee/experiments/tsm-trainer_001/tsm-trainer/outputs/eo-v4-120M-toto_2080ti/best_checkpoints"
)
CKPTS=(); STAGE="all"; DEPTH=1; BATCH=32; OUT=""; ALPHA=""
BENCHMARKS=""; MAX_TASKS=""; MAX_ITEMS=""; MAX_WINDOWS=""
TASK_SUBSET=""; FEV_DATA=""; GIFT_DATA=""; DRY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)        CKPTS+=("$2");   shift 2 ;;
        --repo)        REPO="$2";       shift 2 ;;
        --threads)     THREADS="$2";   shift 2 ;;
        --stage)       STAGE="$2";      shift 2 ;;
        --depth)       DEPTH="$2";      shift 2 ;;
        --batch-size)  BATCH="$2";      shift 2 ;;
        --alpha)       ALPHA="$2";      shift 2 ;;
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
# CPU CAP — EXPORTED BEFORE PYTHON STARTS, which is the only time it works for
# OpenMP/BLAS: those size their pools when the library is first loaded, so a
# value set after `import torch` is ignored. finar_cpu.py handles what CAN be
# set at runtime (torch, pyarrow, fev). multiprocessing.cpu_count() also ignores
# cgroup quotas, so on a node reporting 255 cores while allocating ~29 the
# uncapped pools size to 255 and the job is killed rather than merely slow.
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_MAX_THREADS="${THREADS}"
export TOKIO_WORKER_THREADS="${THREADS}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${THREADS}"
export TOKENIZERS_PARALLELISM=false

note() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

if [[ "${STAGE}" == "eval" || "${STAGE}" == "all" ]]; then
    for CKPT in "${CKPTS[@]}"; do
        TAG="$(basename "${CKPT}")"
        # The alpha grid is part of the identity of a run, not just of its
        # rows: keyed on the model alone, a sweep requested after a default
        # --alpha 0 run was skipped with "csv exists" and never happened.
        if [[ -f "${OUT}/${TAG}.csv" ]] \
           && "${PY}" -c "import json,sys
m=json.load(open(sys.argv[1])); want=[float(x) for x in sys.argv[2].split()]
sys.exit(0 if set(want) <= set(m.get('alpha', [0.0])) else 1)" \
                "${OUT}/${TAG}.meta.json" "${ALPHA:-0}" 2>/dev/null; then
            note "SKIP ${CKPT} — ${OUT}/${TAG}.csv already covers alpha ${ALPHA:-0}"
            continue
        fi
        note "=== ${CKPT}"
        # One model per process: a failure on one must not cost the others, and
        # each holds a whole checkpoint in VRAM.
        ${DRY} "${PY}" "${HERE}/run_eval.py" \
            --model-path "${CKPT}" --repo "${REPO}" --out "${OUT}" \
            --coe-eval-depth "${DEPTH}" --batch-size "${BATCH}" \
            ${ALPHA:+--alpha ${ALPHA}} \
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
