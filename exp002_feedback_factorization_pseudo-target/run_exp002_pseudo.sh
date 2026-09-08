#!/usr/bin/env bash
# ===========================================================================
# FiNAR Experiment 2 (pseudo-target) — which half of the feedback earns it?
# ===========================================================================
# FiNAR feeds the whole intermediate forecast back into the next pass. This
# asks which half earns the improvement — the model's own trajectory
# (self-regression) or its forecast of the OTHER variates (cross-variate
# regression) — by running ONE checkpoint under three feedback regimes:
#
#   full        every variate is fed back (stock EO v4 evaluation)
#   self_only   the designated target is fed back; the other variates are not
#   cov_only    the other variates are fed back; the designated target is not
#
# and comparing each against iteration 1, which is the no-feedback baseline and
# is identical in all three by construction.
#
# ---------------------------------------------------------------------------
# WHY "PSEUDO-TARGET", AND WHY NOT THE COVARIATE SUBSET
# ---------------------------------------------------------------------------
# The covariate-subset version (../exp002_feedback_factorization) runs where
# fev-bench names the target and covariate columns. 9 of its 42 rows are
# MULTI-TARGET — 7 of the 12 past_only rows, which is the only stratum where
# self_only suppresses anything — and self_only feeds back EVERY target of such
# a task. On those rows "self" already contains cross-variate feedback, so the
# split it reports is not the split it names.
#
# This version evaluates on suites where every variate is a target and there
# are no covariates at all:
#
#   fev_mul    fev-bench's `multivariate` subset (>1 target, no dynamic covs)
#   gift_mul   GIFT-Eval HF, multivariate tasks fed whole (group attention on)
#
# and DESIGNATES the first variate of each group as the target, the rest as
# pseudo-covariates. "Self" is then exactly one variate and "cross" exactly the
# others, on every task.
#
# THE DESIGNATION CANNOT MOVE. It is positional — variate 0 of each group, in
# the array order the data is stored in — so it is the same series in every
# regime and at every recursion depth. Each run writes designation.json, and
# run_eval.py REFUSES to finish if the three regimes disagree about what they
# designated.
#
# ---------------------------------------------------------------------------
# USAGE — ON THE REMOTE A100, OVER SSH (not as a submitted job)
# ---------------------------------------------------------------------------
#   ssh <a100-host>
#   cd /group-volume/workspace/mun-hak.lee/experiments/finar_001/finar/exp002_feedback_factorization_pseudo-target
#
#   # everything: score three regimes on both benchmarks, then table + figure
#   bash run_exp002_pseudo.sh --ckpt /path/to/eo-v4-K2/best_checkpoints \
#                             --repo /group-volume/.../tsm-trainer_001/tsm-trainer
#
#   # one benchmark only
#   bash run_exp002_pseudo.sh --ckpt <ckpt> --benchmarks fev_mul
#
#   # table and figure only, from a finished run
#   bash run_exp002_pseudo.sh --stage table --ckpt <ckpt>
#
# Long runs: detach, since an ssh drop kills the job.
#   nohup bash run_exp002_pseudo.sh --ckpt <ckpt> > exp002_pseudo.log 2>&1 &
#   tail -f exp002_pseudo.log
#
# Re-running is safe: a (benchmark, scenario) whose results.csv exists is
# skipped, so an interrupted sweep resumes where it stopped. A re-run with a
# DIFFERENT checkpoint into the same --out is refused rather than silently
# mixing arms — manifest.json records what the directory was built from.
#
# ---------------------------------------------------------------------------
# OPTIONS
# ---------------------------------------------------------------------------
#   --ckpt PATH         EO v4 checkpoint to score. Required.
#   --repo PATH         tsm-trainer checkout to IMPORT from. Read-only.
#   --stage eval|table|all      default: all
#   --benchmarks "fev_mul gift_mul"   default: both
#   --scenarios "full self_only cov_only"  default: all three
#   --depth N           --coe-eval-depth, default 2 (must be >= 2)
#   --batch-size N      default 64
#   --out PATH          results dir, default ./results/<ckpt basename>
#   --fev-data PATH     fev parquet root (default: the local cache)
#   --gift-data PATH    GIFT-Eval root (default: the local cache)
#   --task-subset "i n" score only task slice i of n. GIFT-Eval does not fit
#                       in 23 GB in one pass on these hosts; the slice is
#                       recorded in manifest.json, since two runs are only
#                       comparable if they scored the same tasks.
#   --dry-run           print the commands without running them
#
# ---------------------------------------------------------------------------
# WHAT COMES OUT
# ---------------------------------------------------------------------------
#   <out>/manifest.json                     checkpoint + depth the arms share
#   <out>/<bench>/<scenario>/results.csv    per-task scores, every depth
#   <out>/<bench>/<scenario>/designation.json   what was designated, per task
#   <out>/exp002_pseudo_table.csv           iter1 vs iter2 per regime
#   <out>/exp002_pseudo_improvement.png     the same as grouped bars
#
# ---------------------------------------------------------------------------
# DEPTH — READ THIS BEFORE BLAMING THE TABLE
# ---------------------------------------------------------------------------
# This needs a checkpoint whose TRAINED depth is >= 2, and the trained depth is
# `coe_train_depth_max` PLUS the grad-free warm-up — not coe_train_depth_max
# alone. A checkpoint at K=1 with init_warmup_depth_max >= 1 has been trained
# on inputs a pass already refined and qualifies; the same checkpoint with
# init_warmup_depth_max null does NOT, because the warm-up is inert at K=1 even
# though init_from_repeat_prediction reads true. run_eval.py computes this and
# says which case it found.
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/group-volume/workspace/mun-hak.lee/experiments/tsm-trainer_001/tsm-trainer}"
CKPT=""; STAGE="all"; DEPTH=2; BATCH=64; OUT=""
BENCHMARKS=""; SCENARIOS=""; FEV_DATA=""; GIFT_DATA=""; TASK_SUBSET=""; DRY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)        CKPT="$2";       shift 2 ;;
        --repo)        REPO="$2";       shift 2 ;;
        --stage)       STAGE="$2";      shift 2 ;;
        --depth)       DEPTH="$2";      shift 2 ;;
        --batch-size)  BATCH="$2";      shift 2 ;;
        --out)         OUT="$2";        shift 2 ;;
        --benchmarks)  BENCHMARKS="$2"; shift 2 ;;
        --scenarios)   SCENARIOS="$2";  shift 2 ;;
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
[[ -n "${CKPT}" ]] || { echo "--ckpt is required" >&2; exit 2; }
PY="${PYTHON:-${REPO}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY="$(command -v python3)"
[[ -x "${PY}" ]] || { echo "no python found (tried ${REPO}/.venv/bin/python)" >&2; exit 1; }

[[ -n "${OUT}" ]] || OUT="${HERE}/results/$(basename "${CKPT}")"
note() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

if [[ "${STAGE}" == "eval" || "${STAGE}" == "all" ]]; then
    note "=== scoring ${CKPT} at depths 1..${DEPTH} -> ${OUT}"
    ${DRY} "${PY}" "${HERE}/run_eval.py" \
        --model-path "${CKPT}" --repo "${REPO}" --out "${OUT}" \
        --coe-eval-depth "${DEPTH}" --batch-size "${BATCH}" \
        ${BENCHMARKS:+--benchmarks ${BENCHMARKS}} \
        ${SCENARIOS:+--scenarios ${SCENARIOS}} \
        ${FEV_DATA:+--fev-data "${FEV_DATA}"} \
        ${GIFT_DATA:+--gift-data "${GIFT_DATA}"} \
        ${TASK_SUBSET:+--task-subset-index ${TASK_SUBSET% *} \
                       --num-task-subsets ${TASK_SUBSET#* }}
fi

if [[ "${STAGE}" == "table" || "${STAGE}" == "all" ]]; then
    note "=== table + figure -> ${OUT}"
    ${DRY} "${PY}" "${HERE}/build_table.py" "${OUT}" --iters 1 "${DEPTH}"
fi

note "done. results under ${OUT}"
