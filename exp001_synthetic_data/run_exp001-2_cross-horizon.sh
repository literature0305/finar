#!/usr/bin/env bash
# ===========================================================================
# FiNAR Experiment 1-2 — cross-horizon dependency x horizon length, per iteration
# ===========================================================================
# Does EO v4's iterative refinement help more when the forecast target carries
# more dependency ACROSS the horizon, and does that change as the horizon grows?
#
#   12 tasks = 4 horizons (16 / 100 / 400 / 1000)
#            x 3 dependency regimes (high / mid / low)
#
# scored at every recursion depth 1..N, so the answer is a surface over
# (horizon, regime, iteration) rather than a single number.
#
# ---------------------------------------------------------------------------
# WHY A NEW CORPUS AND NOT THE PUBLISHED ONE
# ---------------------------------------------------------------------------
# /group-volume/ts-dataset/cross_horizon_length holds the observed context
# byte-identical across all twelve subsets — 8192 series, 2048 steps. The MODEL
# does not see 2048 of them. aed/model_base.py::append_forecast_region takes the
# forecast region out of the same window:
#
#     forecast_len = ceil(prediction_length / patch_size) * patch_size
#     max_ctx      = context_length - forecast_len
#
# so at context_length 2048 / patch 16 the history actually reaching the encoder
# is 2032 / 1936 / 1648 / 1040 for H = 16 / 100 / 400 / 1000. The horizon axis is
# collinear with "how much history the model got", and a MASE that rises with H
# cannot be attributed to the horizon.
#
# build_cross_horizon_trim.py therefore re-cuts every subset to 1040 observed
# steps — the value the LONGEST horizon leaves — so no horizon truncates and the
# sweep varies the horizon alone. 1040, not 1048: forecast_len rounds UP to a
# patch multiple, ceil(1000/16)*16 = 1008.
#
# Not preserved: the source README's oracle / Chronos-2 MASE, which were
# measured against 2048 steps and a denominator computed from them. Numbers here
# are comparable to each other, not to that table.
#
# ---------------------------------------------------------------------------
# DEPTH BEYOND TRAINING IS THE QUESTION
# ---------------------------------------------------------------------------
# --depth may exceed the checkpoint's coe_train_depth_max. EOPipeline.report_depth
# is max(trained, requested), so a K=2 checkpoint asked for 4 reports repeat1..4.
# Those rows are recorded and FLAGGED (`beyond_trained_depth` in the csv, a red
# line in the figure) rather than hidden or refused.
#
# ---------------------------------------------------------------------------
# USAGE — ON THE REMOTE A100, OVER SSH
# ---------------------------------------------------------------------------
#   cd /group-volume/workspace/mun-hak.lee/experiments/finar_001/finar/exp001_synthetic_data
#
#   # everything: build the corpus if absent, score depths 1..4, table + figure
#   bash run_exp001-2_cross-horizon.sh --ckpt /path/to/eo-v4/best_checkpoints
#
#   bash run_exp001-2_cross-horizon.sh --ckpt <ckpt> --depth 6
#   bash run_exp001-2_cross-horizon.sh --stage build          # corpus only
#   bash run_exp001-2_cross-horizon.sh --stage table --ckpt <ckpt>
#
# ---------------------------------------------------------------------------
# MOVING THIS TO THE REMOTE A100 — WHAT HAS TO GO WITH IT
# ---------------------------------------------------------------------------
# The corpus is self-contained: `save_to_disk` Arrow, 536 MB, no path inside it
# points back at the source. Copying ONE directory is enough to evaluate:
#
#   rsync -a /group-volume/ts-dataset/cross_horizon_length_trim_equal_observed/ \
#            <a100>:/group-volume/ts-dataset/cross_horizon_length_trim_equal_observed/
#
# The PUBLISHED corpus (cross_horizon_length) does NOT need to travel. It is
# read only by --stage build, and only for subsets that are missing; with the
# trimmed corpus in place every subset is skipped and the source is never
# opened. `--stage all` is therefore safe there.
#
# If the remote path differs, pass --data-root; the 12-task yaml is regenerated
# into --out on every run from that root, so there is no stale path to fix.
#
# Then, on the A100:
#
#   ssh <a100-host>
#   cd .../finar_001/finar/exp001_synthetic_data
#   nohup bash run_exp001-2_cross-horizon.sh \
#       --ckpt /path/to/eo-v4/best_checkpoints \
#       --repo /group-volume/.../tsm-trainer_001/tsm-trainer \
#       --depth 4 --batch-size 64 > exp001-2.log 2>&1 &
#   tail -f exp001-2.log
#
# Runtime reference: 12 tasks x 8192 series at depth 4 took 810 s on a 12 GB
# card at --batch-size 32 (forecast 701 s, metrics 5 s), peak well under 2 GB.
# An A100 40G should take --batch-size 256 comfortably.
#
# Long runs detach, since an ssh drop kills the job:
#   nohup bash run_exp001-2_cross-horizon.sh --ckpt <ckpt> > exp001-2.log 2>&1 &
#
# ---------------------------------------------------------------------------
# OPTIONS
# ---------------------------------------------------------------------------
#   --ckpt PATH         EO v4 checkpoint. Required except for --stage build.
#   --repo PATH         tsm-trainer checkout to IMPORT from. Read-only.
#   --stage build|eval|table|all      default: all
#   --depth N           --coe-eval-depth (default: 4). Reports every depth 1..N.
#                       Values above the checkpoint's coe_train_depth_max are the
#                       EXTRAPOLATION arm and are flagged, not refused.
#   --batch-size N      default 32
#   --data-root PATH    trimmed corpus root (default: the standard location)
#   --src PATH          published corpus to cut from. Only read by --stage build,
#                       and only for subsets that do not exist yet.
#   --out PATH          results dir, default ./results_cross_horizon/<ckpt name>
#   --force-data        re-cut subsets that already exist
#   --dry-run           print the commands without running them
#
# ---------------------------------------------------------------------------
# WHAT COMES OUT
# ---------------------------------------------------------------------------
#   <out>/results.csv                  one row per task, repeat<r>_MASE/WQL
#   <out>/manifest.json                checkpoint, depths, window, observed report
#   <out>/tasks.yaml                   the 12-task benchmark, generated from disk
#   <out>/cross_horizon_table.csv      horizon x regime x iteration, long format
#   <out>/cross_horizon_observed.csv   horizon vs observed length, both corpora
#   <out>/cross_horizon_heatmaps.png   regime x iteration, horizon x iteration
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/group-volume/workspace/mun-hak.lee/experiments/tsm-trainer_001/tsm-trainer}"
CKPT=""; STAGE="all"; DEPTH=4; BATCH=32; OUT=""; DRY=""
DATA="/group-volume/ts-dataset/cross_horizon_length_trim_equal_observed"
SRC="/group-volume/ts-dataset/cross_horizon_length"
FORCE_DATA=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)         CKPT="$2";  shift 2 ;;
        --repo)         REPO="$2";  shift 2 ;;
        --stage)        STAGE="$2"; shift 2 ;;
        --depth)        DEPTH="$2"; shift 2 ;;
        --batch-size)   BATCH="$2"; shift 2 ;;
        --data-root)    DATA="$2";  shift 2 ;;
        --src)          SRC="$2";   shift 2 ;;
        --out)          OUT="$2";   shift 2 ;;
        --force-data)   FORCE_DATA="--force"; shift ;;
        --dry-run)      DRY="echo [dry]"; shift ;;
        # To the end of the header, not a hard-coded line number: the header IS
        # the documentation and a fixed range truncates it as it grows.
        -h|--help)      sed -n '2,${/^set -/q;p;}' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -d "${REPO}" ]] || { echo "--repo ${REPO} is not a directory" >&2; exit 2; }
case "${STAGE}" in build|eval|table|all) ;; *)
    echo "--stage must be build, eval, table or all (got ${STAGE})" >&2; exit 2 ;;
esac
if [[ "${STAGE}" != "build" ]]; then
    [[ -n "${CKPT}" ]] || { echo "--ckpt is required for --stage ${STAGE}" >&2; exit 2; }
fi
PY="${PYTHON:-${REPO}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY="$(command -v python3)"
[[ -x "${PY}" ]] || { echo "no python found (tried ${REPO}/.venv/bin/python)" >&2; exit 1; }

[[ -n "${OUT}" ]] || OUT="${HERE}/results_cross_horizon/$(basename "${CKPT:-none}")"
note() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

if [[ "${STAGE}" == "build" || "${STAGE}" == "all" ]]; then
    note "=== equal-observed corpus -> ${DATA}"
    ${DRY} "${PY}" "${HERE}/build_cross_horizon_trim.py" \
        --src "${SRC}" --dst "${DATA}" ${FORCE_DATA}
fi

if [[ "${STAGE}" == "eval" || "${STAGE}" == "all" ]]; then
    note "=== scoring ${CKPT} at depths 1..${DEPTH} -> ${OUT}"
    ${DRY} "${PY}" "${HERE}/run_eval_cross_horizon.py" \
        --model-path "${CKPT}" --repo "${REPO}" --out "${OUT}" \
        --data "${DATA}" --coe-eval-depth "${DEPTH}" --batch-size "${BATCH}"
fi

if [[ "${STAGE}" == "table" || "${STAGE}" == "all" ]]; then
    note "=== table + heatmaps -> ${OUT}"
    ${DRY} "${PY}" "${HERE}/build_table_cross_horizon.py" "${OUT}"
fi

note "done. results under ${OUT}"
