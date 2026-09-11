#!/usr/bin/env bash
# ===========================================================================
# FiNAR Experiment 7 — does RESIDUAL REFINEMENT improve iTransformer?
# ===========================================================================
# iTransformer forecasts the whole horizon in ONE pass: each variate's lookback
# becomes a token, variates attend to each other, and a linear head emits all
# `pred_len` steps at once. EO v4 (tsm-trainer) does the opposite — it runs one
# weight-shared encoder N times, writing each pass's forecast back into the
# positions it has to predict, so a later pass starts from a better guess and
# only has to correct it.
#
# This experiment puts EO v4's chain of encoders on iTransformer and asks
# whether the correction buys anything on the paper's own benchmark. The
# comparison is against a baseline that REPRODUCES the paper, not against a
# reimplementation that merely resembles it: `precheck.py` asserts, before any
# training starts, that this model and the official `thuml/iTransformer` one
# accept the same weights and return bit-identical forecasts, and that every
# split matches the official loader window for window.
#
# Nothing here imports tsm-trainer. The option names are EO v4's so the two
# experiments can be read side by side; the code is standalone.
#
# ---------------------------------------------------------------------------
# HOW THE REFINEMENT WORKS
# ---------------------------------------------------------------------------
# iTransformer has no masked value channel to write into — a variate's token IS
# its whole window — so the write-back becomes a widened window:
#
#     pass i     token_v = Linear([ lookback_v (96) ; A_{i-1,v} (pred_len) ])
#     A_0 = 0,   A_i = the forecast reported by pass i
#
# With `--coe-residual true` (the default, and the point of the experiment) each
# pass emits a DELTA and reports `A_{i-1} + delta_i`. Because `A_0 = 0`, one
# pass is bit-identical to the non-residual model, and — with the feedback
# slot's weights zeroed — the whole chain collapses onto the baseline exactly.
# `precheck.py` pins both, and pins that with real weights the chain DOES move,
# which is what separates "the refinement did not help" from "the refinement
# never ran".
#
# ---------------------------------------------------------------------------
# USAGE — REPRODUCING THE PAPER
# ---------------------------------------------------------------------------
# Table 10 of arXiv:2310.06625v3 is nine datasets x four horizons at lookback
# 96. Every default here comes from the official scripts (see paper.py), so the
# paper's configuration is just the dataset and the horizon:
#
#   bash prepare_data.sh --with-reference          # once: data + the checkout
#   bash train.sh --dataset ETTh1 --pred-len 96    # -> MSE 0.386, MAE 0.405
#
#   # the whole published table for one dataset
#   bash train.sh --dataset ETTh1 --pred-len "96 192 336 720"
#
#   # several datasets, one after another
#   bash train.sh --dataset "ETTh1 ETTh2 ETTm1 ETTm2 Weather Solar" \
#                 --pred-len "96 192 336 720"
#
# Each run prints its own verdict against the published cell, e.g.
#   vs paper: MATCH — paper MSE 0.386 MAE 0.405 | got MSE 0.3868 (+0.0008) ...
# A run outside the tolerance in paper.py is reported as a MISS, not rounded in.
#
# ---------------------------------------------------------------------------
# USAGE — THE TWO TRAINING SCENARIOS THIS EXPERIMENT COMPARES
# ---------------------------------------------------------------------------
# Scenario A — BASELINE. The paper's model, unchanged. This is the number the
# refinement has to beat, and the number the precheck says is the paper's.
#
#   bash train.sh --dataset ETTh1 --pred-len 96 --refinement off
#
# Scenario B — RESIDUAL REFINEMENT. The same model, same data, same optimizer,
# with EO v4's chain of encoders. `--train-depth 3` draws N ~ Uniform{1,2,3}
# per step so one checkpoint serves every depth; `--depth 3` is what inference
# uses, and it MAY exceed the training depth (that is the test-time scaling EO
# v4 is for).
#
#   bash train.sh --dataset ETTh1 --pred-len 96 --refinement on \
#                 --train-depth 3 --depth 3 --eval-depths "1 2 3 4"
#
# Run both and table them:
#
#   for R in off on; do
#       bash train.sh --dataset ETTh1 --pred-len "96 192 336 720" \
#                     --refinement $R --train-depth 3 --depth 3 \
#                     --eval-depths "1 2 3 4"
#   done
#   bash eval.sh --stage table
#
# Ablations, all in EO v4's vocabulary:
#   --coe-residual false     the chain re-predicts instead of correcting
#   --coe-bottleneck false   chain the encoder in HIDDEN space; nothing is fed
#                            back (EO v4's no-bottleneck ablation)
#   --coe-backprop all       unroll BPTT through every pass (default: only the
#                            last pass is graphed)
#   --coe-internal-loss true supervise every pass (needs --coe-backprop all)
#   --coe-future-marks false zero-fill the fed-back window's timestamps, which
#                            makes a depth-1 forward bit-identical to baseline
#
# ---------------------------------------------------------------------------
# USAGE — ON THE REMOTE A100
# ---------------------------------------------------------------------------
# Over ssh, like the other experiments:
#   ssh <a100-host>
#   cd .../finar/exp007_itransformer_residual_refinement
#   bash prepare_data.sh                       # once, onto the shared volume
#   nohup bash train.sh --dataset ETTh1 --pred-len "96 192 336 720" \
#         --refinement on --train-depth 3 --depth 3 > exp007.log 2>&1 &
#
# Or as a submitted job (one job runs every cell, then evaluates each):
#   bash train.sh --mode job --dataset ETTh1 --pred-len "96 192 336 720" \
#                 --refinement on --train-depth 3 --depth 3 --ngpu 1
#   bash train.sh --mode job ... --dry-run     # print the ssub command only
#
# ONLY Traffic and ECL are large (862 and 321 variates, batch 16). The rest fit
# a single GPU comfortably; --ngpu 1 is the default for that reason.
#
# ---------------------------------------------------------------------------
# OPTIONS
# ---------------------------------------------------------------------------
#   --dataset "A B"     one or more of ETTh1 ETTh2 ETTm1 ETTm2 ECL Traffic
#                       Weather Exchange Solar. Required.
#   --pred-len "96 192" horizons to train, one run each. Default: 96
#   --seq-len N         lookback. Default 96 — the paper's protocol throughout
#   --data-root PATH    dataset root (default /group-volume/ts-dataset/ltsf)
#   --out PATH          run directories (default ./runs)
#   --refinement on|off apply the EO v4 chain of encoders. Default: off
#   --train-depth K     coe_train_depth_max: training draws N ~ U{1..K}
#   --depth N           coe_eval_depth: passes at inference. May exceed K
#   --eval-depths "1 2" score these depths when training finishes (one forward)
#   --coe-residual B    each pass emits a delta (default true)
#   --coe-bottleneck B  feed the forecast back through the embedding
#                       (default true); false = hidden-state chain
#   --coe-stochastic-repeat B   draw N per step, else always K (default true)
#   --coe-backprop last|all     graph only the last pass, or unroll (last)
#   --coe-internal-loss B       supervise every pass (needs --coe-backprop all)
#   --coe-future-marks B        real timestamps in the fed-back window (true)
#   --epochs N / --batch-size N / --lr F / --seed N
#                       override the official setting for this cell
#   --num-workers N     CPU cap for the run. Default: read from the scheduler
#                       affinity mask and the cgroup quota, so a quota'd node
#                       configures itself. A value above the allocation is
#                       clamped, not obeyed.
#   --loader-workers N  DataLoader worker processes (default 0). The dataset
#                       is an in-memory array, so workers add IPC and nothing
#                       else — 0 measured fastest on both ETTh1 and Weather.
#                       Any workers are reserved OUT of the parent's thread
#                       pool: --num-workers is a budget for the process group,
#                       not for one pool.
#   --device cuda|cpu   default: cuda when available
#   --run-name NAME     override the run directory name. Rarely needed: the
#                       name already carries a digest of whatever was
#                       overridden, so two runs cannot collide
#   --reference PATH    official checkout for the precheck
#                       (default ./reference/iTransformer). MISSING IT IS AN
#                       ERROR: the baseline's agreement with the paper is the
#                       premise of the whole comparison
#   --no-reference      train anyway, with the official-model and data
#                       comparisons NOT run. Explicit on purpose
#   --skip-precheck     train without checking at all. Say why in the log.
#   --mode local|job    run here, or submit via ssub (default local)
#   --ngpu N / --gpu-type A100|H100|2080ti / --job-name S / --priority N /
#   --exp-id N / --image S      job mode only
#   --dry-run           print the commands without running them
#
# ---------------------------------------------------------------------------
# WHAT COMES OUT
# ---------------------------------------------------------------------------
#   <out>/<dataset>_<seq>_<pred>_<variant>[_<digest>]/
#                       the digest appears only when something was overridden,
#                       so two runs differing in a seed cannot share a directory
#       config.json      the exact model + optimizer settings, for reloading
#       checkpoint.pt    the best epoch by validation loss
#       train_log.json   per-epoch train/val loss and the best epoch
#       metrics.json     test MSE/MAE, per depth, and the verdict vs the paper
#   <out>/exp007_table.csv, exp007_refinement.png   (eval.sh --stage table)
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/group-volume/ts-dataset/ltsf}"
THREADS="${THREADS:-}"   # empty = derive from the allocation
ORIG=("$@")          # forwarded verbatim in job mode; see PASSTHRU below
DATASETS=""; PRED_LENS="96"; SEQ_LEN="96"; OUT=""; RUN_NAME=""
REFINEMENT="off"; COE=()   # chain settings, already in train.py's flag names
EPOCHS=""; BATCH=""; LR=""; SEED=""; LOADER_WORKERS=""; DEVICE=""
REFERENCE=""; SKIP_PRECHECK=""; NO_REFERENCE=""; MODE="local"; DRY=""
NGPU="1"; GPU_TYPE="A100"; JOB_NAME=""; PRIORITY=""; EXP_ID=""; IMAGE=""

#: Flags that configure the SUBMISSION, not the run. Job mode forwards
#: everything else to itself inside the container, so these are the only ones
#: it has to strip. Split by arity, because stripping a valueless flag as if
#: it took one would eat the argument after it.
JOB_ONLY=" --mode --ngpu --gpu-type --job-name --priority --exp-id --image "
JOB_ONLY_SWITCHES=" --dry-run "

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASETS="${DATASETS} $2"; shift 2 ;;
        --pred-len)      PRED_LENS="$2";     shift 2 ;;
        --seq-len)       SEQ_LEN="$2";       shift 2 ;;
        --data-root)     DATA_ROOT="$2";     shift 2 ;;
        --out)           OUT="$2";           shift 2 ;;
        --refinement)    REFINEMENT="$2";    shift 2 ;;
        --run-name)      RUN_NAME="$2";      shift 2 ;;
        # The chain settings go into ONE array under the names train.py uses,
        # so adding a knob is one line here instead of five. train.py returns
        # them to their defaults for a baseline run and says which it ignored,
        # so they are forwarded unconditionally and `python train.py` behaves
        # the same way as this launcher.
        --train-depth)   COE+=(--coe-train-depth-max "$2"); shift 2 ;;
        --depth)         COE+=(--coe-eval-depth "$2");      shift 2 ;;
        # Unquoted on purpose: argparse takes nargs="+" here, so "1 2 3" has to
        # arrive as three arguments.
        --eval-depths)   COE+=(--eval-depths $2);           shift 2 ;;
        --coe-residual|--coe-bottleneck|--coe-stochastic-repeat|\
        --coe-backprop|--coe-internal-loss|--coe-future-marks)
                         COE+=("$1" "$2");                  shift 2 ;;
        --epochs)        EPOCHS="$2";        shift 2 ;;
        --batch-size)    BATCH="$2";         shift 2 ;;
        --lr)            LR="$2";            shift 2 ;;
        --seed)          SEED="$2";          shift 2 ;;
        --num-workers)   THREADS="$2";       shift 2 ;;
        --loader-workers) LOADER_WORKERS="$2"; shift 2 ;;
        --device)        DEVICE="$2";        shift 2 ;;
        --reference)     REFERENCE="$2";     shift 2 ;;
        --no-reference)  NO_REFERENCE="1";   shift ;;
        --skip-precheck) SKIP_PRECHECK="1";  shift ;;
        --mode)          MODE="$2";          shift 2 ;;
        --ngpu)          NGPU="$2";          shift 2 ;;
        --gpu-type)      GPU_TYPE="$2";      shift 2 ;;
        --job-name)      JOB_NAME="$2";      shift 2 ;;
        --priority)      PRIORITY="$2";      shift 2 ;;
        --exp-id)        EXP_ID="$2";        shift 2 ;;
        --image)         IMAGE="$2";         shift 2 ;;
        --dry-run)       DRY="echo [dry]";   shift ;;
        # To the end of the header, not a fixed line count: the header IS the
        # documentation and a hard-coded range truncates it as it grows.
        -h|--help)       sed -n '2,${/^set -/q;p;}' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

DATASETS="${DATASETS# }"
[[ -n "${DATASETS}" ]] || { echo "--dataset is required (see --help)" >&2; exit 2; }
case "${MODE}" in local|job) ;; *)
    echo "--mode must be local or job (got ${MODE})" >&2; exit 2 ;;
esac
case "${REFINEMENT}" in on|off|true|false) ;; *)
    echo "--refinement must be on or off (got ${REFINEMENT})" >&2; exit 2 ;;
esac
case "${REFINEMENT}" in on|true) REFINE="true" ;; *) REFINE="false" ;; esac

[[ -n "${OUT}" ]] || OUT="${HERE}/runs"
[[ -n "${REFERENCE}" ]] || REFERENCE="${HERE}/reference/iTransformer"
# CPU cap: one definition, sourced. See finar_cpu.sh / finar_cpu.py.
. "$(dirname "${HERE}")/finar_cpu.sh"
# Interpreter + note(): one definition, sourced. See _common.sh.
. "${HERE}/_common.sh"

# ---- job mode: hand the whole thing to one allocation and stop here --------
if [[ "${MODE}" == "job" ]]; then
    # This launcher's OWN arguments, minus the submission flags, are what the
    # container runs. Re-declaring each of them in submit_job.py meant every
    # new flag had to be spelled in five places, and a missed one failed only
    # inside a submitted container, hours later.
    PASSTHRU=(); i=0
    while [[ ${i} -lt ${#ORIG[@]} ]]; do
        A="${ORIG[$i]}"
        if [[ "${JOB_ONLY}" == *" ${A} "* ]]; then
            i=$((i + 2))
        elif [[ "${JOB_ONLY_SWITCHES}" == *" ${A} "* ]]; then
            i=$((i + 1))
        else
            PASSTHRU+=("${A}"); i=$((i + 1))
        fi
    done
    note "submitting: ${DATASETS} x ${PRED_LENS}, refinement=${REFINE}"
    # No ${DRY} prefix here: submit_job.py has its own --dry-run, which prints
    # the ssub command instead of echoing the call that would print it.
    "${PY}" "${HERE}/submit_job.py" \
        --ngpu "${NGPU}" --gpu-type "${GPU_TYPE}" \
        ${JOB_NAME:+--job-name "${JOB_NAME}"} \
        ${PRIORITY:+--priority "${PRIORITY}"} \
        ${EXP_ID:+--exp-id "${EXP_ID}"} ${IMAGE:+--image "${IMAGE}"} \
        ${DRY:+--dry-run} \
        -- ${PASSTHRU[@]+"${PASSTHRU[@]}"}
    exit 0
fi

# ---- precheck: is this the paper's model, and does the refinement run? -----
if [[ -n "${SKIP_PRECHECK}" ]]; then
    note "SKIPPING the precheck (--skip-precheck): the baseline's agreement with the paper and the refinement's reachability are UNVERIFIED for this run"
else
    note "=== precheck"
    # precheck.py owns the policy: a checkout that is absent or unusable is a
    # SKIP, and any SKIP exits non-zero unless --no-reference says the gap is
    # accepted. Re-deciding that here gave two gates that disagreed on an
    # existing-but-wrong --reference path.
    ${DRY} "${PY}" "${HERE}/precheck.py" \
        --data-root "${DATA_ROOT}" --seq-len "${SEQ_LEN}" \
        --reference "${REFERENCE}" \
        ${NO_REFERENCE:+--no-reference} \
        ${THREADS:+--num-workers "${THREADS}"}
fi

# ---- train ----------------------------------------------------------------
mkdir -p "${OUT}"
for DS in ${DATASETS}; do
    for H in ${PRED_LENS}; do
        note "=== ${DS} / pred_len ${H} / refinement ${REFINE}"
        ${DRY} "${PY}" "${HERE}/train.py" \
            --dataset "${DS}" --pred-len "${H}" --seq-len "${SEQ_LEN}" \
            --data-root "${DATA_ROOT}" --out "${OUT}" \
            --refinement "${REFINE}" \
            ${THREADS:+--num-workers "${THREADS}"} \
            ${RUN_NAME:+--run-name "${RUN_NAME}"} \
            ${COE[@]+"${COE[@]}"} \
            ${EPOCHS:+--epochs "${EPOCHS}"} ${BATCH:+--batch-size "${BATCH}"} \
            ${LR:+--learning-rate "${LR}"} ${SEED:+--seed "${SEED}"} \
            ${LOADER_WORKERS:+--loader-workers "${LOADER_WORKERS}"} \
            ${DEVICE:+--device "${DEVICE}"} \
            || note "FAILED: ${DS}/${H} (continuing with the rest)"
    done
done

note "=== table -> ${OUT}"
${DRY} "${PY}" "${HERE}/build_table.py" "${OUT}"
note "done. runs under ${OUT}"
