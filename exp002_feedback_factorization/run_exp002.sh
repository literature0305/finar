#!/usr/bin/env bash
# ===========================================================================
# FiNAR Experiment 2 — which half of the feedback earns the refinement?
# ===========================================================================
# FiNAR feeds the whole intermediate forecast back into the next pass. This
# asks whether the gain comes from the model's own target trajectory
# (self-regression) or from its forecast of the covariates (covariate
# regression), by running ONE checkpoint under three feedback regimes:
#
#   full        every variate is fed back (stock EO v4 evaluation)
#   self_only   targets fed back; PAST-ONLY covariates not. Known-future
#               covariates are still given — their forecast region is
#               OBSERVED, not predicted, so it is not feedback at all.
#   cov_only    covariates fed back; every target not.
#
# scored on fev-bench's covariate subset, split three ways (past-only /
# future-known / mixed), at iteration 1 and iteration 2.
#
# ---------------------------------------------------------------------------
# USAGE — ON THE REMOTE A100, OVER SSH (not as a submitted job)
# ---------------------------------------------------------------------------
#   ssh <a100-host>
#   cd /group-volume/workspace/mun-hak.lee/experiments/finar_001/finar/exp002_feedback_factorization
#
#   bash run_exp002.sh --ckpt /path/to/eo-v4-K2/best_checkpoints \
#                      --repo /group-volume/.../tsm-trainer_001/tsm-trainer
#
#   # tables only, from a finished run
#   bash run_exp002.sh --stage table --ckpt <ckpt>
#
# Long runs: detach, since an ssh drop kills the job.
#   nohup bash run_exp002.sh --ckpt <ckpt> > exp002.log 2>&1 &
#   tail -f exp002.log
#
# Re-running is safe: a scenario whose fev_bench.csv exists is skipped, so an
# interrupted sweep resumes where it stopped.
#
# ---------------------------------------------------------------------------
# OPTIONS
# ---------------------------------------------------------------------------
#   --ckpt PATH        EO v4 checkpoint. REQUIRED.
#   --repo PATH        tsm-trainer checkout to IMPORT from. Read-only: this
#   --threads N         CPU threads this run may use. Default 8.
#                       Use 8, 16 or 32; the pools are capped BEFORE
#                       python starts, which is the only time it works
#                       for OpenMP/BLAS.
#                      experiment never writes to it.
#   --stage all|eval|table            default: all
#   --scenarios "full self_only"      default: all three
#   --depth N          --coe-eval-depth, default 2
#   --fev-data PATH    fev parquet cache
#   --out PATH         results dir, default ./results/<ckpt basename>
#   --batch-size N     default 32
#   --dry-run          print the commands without running them
#
# ---------------------------------------------------------------------------
# THE CHECKPOINT MUST SATISFY THREE THINGS, AND IS REFUSED OTHERWISE
# ---------------------------------------------------------------------------
#   coe_bottleneck == True     Without it the passes chain in HIDDEN space and
#                              never write a forecast back, so "feed back only
#                              the targets" has no meaning at all.
#   coe_train_depth_max >= 2   Otherwise iteration 2 is extrapolation past the
#                              training regime rather than refinement — and the
#                              depth columns would be silently absent, because
#                              _repeats_worth_asking returns False as soon as
#                              report_depth() == 1, BEFORE it reads
#                              TSM_FEV_COE_REPEATS.
#   eo_config present          i.e. actually an EO v4 model.
#
# Check a candidate before queueing a long run:
#   python3 -c "import json;e=json.load(open('\$CKPT/config.json'))['eo_config']
#   print(e.get('coe_bottleneck'), e.get('coe_train_depth_max'),
#         e.get('coe_residual'), e.get('feedback_variable_drop_max_ratio'))"
#
# ---------------------------------------------------------------------------
# KNOWN TRAIN/EVAL MISMATCH (accepted, by the work order)
# ---------------------------------------------------------------------------
# feedback_variable_drop_max_ratio is a cap on a COUNT, not a probability: at
# r < 1.0 a group always keeps at least one refined variate (a 7-variate group
# at r=0.9 drops 0..6, never 7). So the model has never seen a group whose
# variates are ALL blank. cov_only (one target dropped) sits inside the training
# distribution; self_only can sit on its edge for covariate-heavy groups. This
# is recorded rather than corrected, and belongs in the writeup.
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/group-volume/workspace/mun-hak.lee/experiments/tsm-trainer_001/tsm-trainer}"
THREADS="${THREADS:-8}"
CKPT=""; STAGE="all"; SCENARIOS=""; DEPTH=2; FEV_DATA=""; OUT=""
BATCH=32; DRY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)       CKPT="$2";      shift 2 ;;
        --repo)       REPO="$2";      shift 2 ;;
        --threads)     THREADS="$2";   shift 2 ;;
        --stage)      STAGE="$2";     shift 2 ;;
        --scenarios)  SCENARIOS="$2"; shift 2 ;;
        --depth)      DEPTH="$2";     shift 2 ;;
        --fev-data)   FEV_DATA="$2";  shift 2 ;;
        --out)        OUT="$2";       shift 2 ;;
        --batch-size) BATCH="$2";     shift 2 ;;
        --dry-run)    DRY="echo [dry]"; shift ;;
        -h|--help)    sed -n '2,80p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "${CKPT}" ]] || { echo "--ckpt is required" >&2; exit 2; }
[[ -d "${REPO}" ]] || { echo "--repo ${REPO} is not a directory" >&2; exit 2; }

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
    note "=== scoring ${CKPT} under the feedback regimes -> ${OUT}"
    ${DRY} "${PY}" "${HERE}/run_eval.py" \
        --num-workers "${THREADS}" \
        --model-path "${CKPT}" --repo "${REPO}" --out "${OUT}" \
        --coe-eval-depth "${DEPTH}" --batch-size "${BATCH}" \
        ${FEV_DATA:+--fev-data "${FEV_DATA}"} \
        ${SCENARIOS:+--scenarios ${SCENARIOS}}
fi

if [[ "${STAGE}" == "table" || "${STAGE}" == "all" ]]; then
    note "=== tables"
    ${DRY} "${PY}" "${HERE}/build_table.py" "${OUT}"
fi

note "done. results under ${OUT}"
