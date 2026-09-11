# Sourced by exp007's launchers. The interpreter and the log line, once.
#
# exp007 imports nothing from tsm-trainer; that checkout is only a convenient
# INTERPRETER with torch/pandas already installed. Set PYTHON to a standalone
# venv (see requirements.txt) to drop even that. The path below is the string
# most likely to change, which is why it is not written in three launchers.
if [[ -z "${PY:-}" ]]; then
    PY="${PYTHON:-}"
fi
if [[ -z "${PY}" ]]; then
    for CAND in "${HERE}/.venv/bin/python" \
                "/group-volume/workspace/mun-hak.lee/experiments/tsm-trainer_001/tsm-trainer/.venv/bin/python" \
                "$(command -v python3 || true)"; do
        [[ -x "${CAND}" ]] && { PY="${CAND}"; break; }
    done
fi
[[ -x "${PY}" ]] || { echo "no python found; set PYTHON=..." >&2; exit 1; }

note() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
