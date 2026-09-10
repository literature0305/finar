# Sourced by every run_exp*.sh. Sets and exports the CPU cap BEFORE python starts.
#
# The python side (finar_cpu.py) caps torch/pyarrow/fev at runtime and reads the
# real allocation, but OpenMP and BLAS size their pools when the library first
# loads — so those must be in the environment before the interpreter runs. That
# is the only part that cannot be done from python, and the only reason this
# file exists.
#
# THREADS may be empty: python then derives the cap from the scheduler affinity
# mask and the cgroup quota, which is the better default. A value that exceeds
# the allocation is clamped there, not here.
if [[ -n "${THREADS}" ]]; then
    [[ "${THREADS}" =~ ^[1-9][0-9]*$ ]] || {
        echo "--num-workers must be a positive integer (got '${THREADS}')" >&2
        exit 2
    }
    # `:-` so a caller's own export wins, matching run_evaluation.sh:1262,
    # run_evaluation_supercom.sh:243 and fast_univariate_csv_eval.sh:18 —
    # parallel_eval.sh relies on that per-shard env passing.
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS}}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${THREADS}}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${THREADS}}"
    export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-${THREADS}}"
    export TOKIO_WORKER_THREADS="${TOKIO_WORKER_THREADS:-${THREADS}}"
    export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-${THREADS}}"
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
