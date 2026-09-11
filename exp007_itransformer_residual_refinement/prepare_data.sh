#!/usr/bin/env bash
# ===========================================================================
# exp007 — fetch and verify the datasets the iTransformer paper reports on
# ===========================================================================
# The paper's Table 10 is nine datasets at lookback 96. None of them ship with
# this repo and none were on the cluster, so this script downloads them and —
# more importantly — CHECKS THEM. A truncated or re-exported csv trains
# happily and produces an MSE that simply is not comparable to the paper's, so
# `verify_data.py` asserts each file's row and variate count before anything
# is trained on it.
#
# SOURCES
#   ETTh1/2, ETTm1/2, electricity, traffic, weather, exchange_rate
#       huggingface.co/datasets/thuml/Time-Series-Library — the authors' own
#       mirror of the files their scripts read.
#   Solar-Energy
#       github.com/laiguokun/multivariate-time-series-data (LSTNet), which is
#       where the benchmark's `solar_AL.txt` originates. The iTransformer
#       README points at a Google Drive / Baidu archive; neither can be
#       fetched non-interactively, which is why these two mirrors are used.
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   # everything, into the shared dataset area (~420 MB)
#   bash prepare_data.sh
#
#   # just the cheap ones, somewhere else
#   bash prepare_data.sh --data-root /scratch/ltsf --datasets "ETTh1 ETTm1"
#
#   # also clone the official repo, which `precheck.py --reference` needs
#   bash prepare_data.sh --with-reference
#
#   # re-check an existing copy without downloading anything
#   bash prepare_data.sh --verify-only
#
# ON THE REMOTE A100: run this ONCE on the shared volume. The datasets are
# small enough to keep beside the other corpora, and every job then points
# --data-root at the same path.
#
# ---------------------------------------------------------------------------
# OPTIONS
# ---------------------------------------------------------------------------
#   --data-root PATH    where the dataset directories live
#                       (default: /group-volume/ts-dataset/ltsf)
#   --datasets "A B"    subset to fetch. Default: every dataset data.py knows
#                       about — ETTh1 ETTh2 ETTm1 ETTm2 ECL Traffic Weather
#                       Exchange Solar
#   --with-reference    also `git clone --depth 1` thuml/iTransformer, which
#                       precheck.py compares this implementation against
#   --reference-dir P   where that clone goes (default: ./reference/iTransformer)
#   --verify-only       check what is already on disk; download nothing
#   --force             re-download even where the file already verifies
#   --strict            treat a sha256 that differs from the reference copy as
#                       a failure, not a note
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/group-volume/ts-dataset/ltsf}"
DATASETS=""          # empty = every dataset data.py knows about
REFERENCE_DIR="${HERE}/reference/iTransformer"
WITH_REFERENCE=""; VERIFY_ONLY=""; FORCE=""; STRICT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)     DATA_ROOT="$2";     shift 2 ;;
        --datasets)      DATASETS="$2";      shift 2 ;;
        --with-reference) WITH_REFERENCE=1;  shift ;;
        --reference-dir) REFERENCE_DIR="$2"; shift 2 ;;
        --verify-only)   VERIFY_ONLY=1;      shift ;;
        --force)         FORCE=1;            shift ;;
        --strict)        STRICT="--strict";  shift ;;
        -h|--help)       sed -n '2,${/^set -/q;p;}' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

THREADS="${THREADS:-}"   # empty = derive from the allocation
# CPU cap: verify_data.py imports data.py, which pulls numpy/pandas/torch —
# uncapped, that is the 255-logical-CPU incident finar_cpu.py exists for.
. "$(dirname "${HERE}")/finar_cpu.sh"
# Interpreter + note(): one definition, sourced. See _common.sh.
. "${HERE}/_common.sh"

HF="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"
LSTNET="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master"

# name | url | post-processing. The DESTINATION is not here: data.py's
# registry already owns which subdirectory and filename each dataset lives in,
# and a second copy of that drifts into downloading to a path the loader never
# reads — which verify_data.py (which does use the registry) then reports as
# [MISSING] for a file that was just fetched.
SPECS=(
    "ETTh1|${HF}/ETT-small/ETTh1.csv|"
    "ETTh2|${HF}/ETT-small/ETTh2.csv|"
    "ETTm1|${HF}/ETT-small/ETTm1.csv|"
    "ETTm2|${HF}/ETT-small/ETTm2.csv|"
    "ECL|${HF}/electricity/electricity.csv|"
    "Traffic|${HF}/traffic/traffic.csv|"
    "Weather|${HF}/weather/weather.csv|"
    "Exchange|${HF}/exchange_rate/exchange_rate.csv|"
    "Solar|${LSTNET}/solar-energy/solar_AL.txt.gz|gunzip"
)

# One python call for all nine destinations, from data.py's own registry.
declare -A DEST_OF
while IFS=$'\t' read -r NAME DEST; do
    DEST_OF["${NAME}"]="${DEST}"
done < <("${PY}" -c '
import sys
import data
root = sys.argv[1]
for name in sorted(data.DATASETS):
    print(f"{name}\t{data.dataset_path(root, name)}")' "${DATA_ROOT}")
[[ ${#DEST_OF[@]} -gt 0 ]] || { echo "could not read the dataset layout from data.py" >&2; exit 1; }
[[ -n "${DATASETS}" ]] || DATASETS="${!DEST_OF[*]}"

wanted() {
    local name="$1"
    for w in ${DATASETS}; do [[ "${w}" == "${name}" ]] && return 0; done
    return 1
}

if [[ -z "${VERIFY_ONLY}" ]]; then
    for SPEC in "${SPECS[@]}"; do
        IFS='|' read -r NAME URL POST <<<"${SPEC}"
        wanted "${NAME}" || continue
        DEST="${DEST_OF[${NAME}]}"
        if [[ -s "${DEST}" && -z "${FORCE}" ]]; then
            note "have ${NAME} (${DEST})"
            continue
        fi
        mkdir -p "$(dirname "${DEST}")"
        note "fetching ${NAME} <- ${URL}"
        # To a temporary name, moved into place only on success: an interrupted
        # download that lands on the final path is the thing `-s` above would
        # then accept forever.
        TMP="${DEST}.part"
        if [[ "${POST}" == "gunzip" ]]; then
            curl -fsSL -o "${TMP}.gz" "${URL}"
            gunzip -f "${TMP}.gz"
        else
            curl -fsSL -o "${TMP}" "${URL}"
        fi
        mv -f "${TMP}" "${DEST}"
    done
fi

if [[ -n "${WITH_REFERENCE}" ]]; then
    if [[ -d "${REFERENCE_DIR}/.git" ]]; then
        note "have the official checkout (${REFERENCE_DIR})"
    else
        note "cloning thuml/iTransformer -> ${REFERENCE_DIR}"
        mkdir -p "$(dirname "${REFERENCE_DIR}")"
        git clone --depth 1 -q https://github.com/thuml/iTransformer.git \
            "${REFERENCE_DIR}"
    fi
fi

note "verifying"
"${PY}" "${HERE}/verify_data.py" --data-root "${DATA_ROOT}" \
    --datasets ${DATASETS} ${STRICT}

note "done. use --data-root ${DATA_ROOT}"
if [[ -n "${WITH_REFERENCE}" ]]; then
    note "and --reference ${REFERENCE_DIR} for precheck.py"
fi
