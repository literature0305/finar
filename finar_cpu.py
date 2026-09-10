#!/usr/bin/env python3
"""One CPU-thread cap for every finar experiment.

WHY THIS EXISTS
---------------
`run_benchmark.py` caps threads in two places: a module-level call that sets the
env-var pools (`OMP/MKL/OPENBLAS/NUMEXPR/TOKIO`) to 8, and a block inside
`main()` that additionally does `torch.set_num_threads`, `pa.set_cpu_count` and
`pa.set_io_thread_count` from `--num-workers`.

Every finar experiment imports `load_forecaster` and drives the adapters itself,
so it gets the first and NOT the second. Measured on an 18-core box with the
intended cap of 8:

    OMP/MKL/OPENBLAS/TOKIO_NUM_THREADS   8      capped
    pyarrow cpu_count / io_thread_count  8      capped
    torch.get_num_threads()             18      NOT capped
    torch.get_num_interop_threads()     18      NOT capped
    fev DEFAULT_NUM_PROC                18      NOT capped

`multiprocessing.cpu_count()` also ignores cgroup quotas, so on a node that
reports 255 logical CPUs while allocating ~29 the uncapped pools size themselves
to 255 — which is how a remote job dies rather than merely running slowly.

ORDER MATTERS, AND HALF OF IT CANNOT BE FIXED FROM PYTHON
---------------------------------------------------------
OpenMP/BLAS size their pools when their library is first loaded, so
`OMP_NUM_THREADS` has to be in the environment BEFORE `import torch`. A python
process that imports torch at module scope has already lost that race. The
launcher scripts therefore export the env vars before starting python, and this
module handles what CAN be set at runtime: torch's own pools, pyarrow's, and
fev's process-pool default.

`limit_cpu` is idempotent and reports what it actually achieved, so a run that
did not get the cap it asked for says so in its log rather than being discovered
by a dead node.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("finar_cpu")

#: What a run uses when nothing says otherwise. Deliberately small: the failure
#: mode of too few threads is a slow run, of too many is a killed one.
DEFAULT_THREADS = 8

_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_MAX_THREADS", "TOKIO_WORKER_THREADS",
             "HF_XET_NUM_CONCURRENT_RANGE_GETS")


def requested_threads(explicit: int | None = None) -> int:
    """The cap this process should use: the flag, else the environment, else 8."""
    if explicit:
        return max(1, int(explicit))
    return max(1, int(os.environ.get("OMP_NUM_THREADS", DEFAULT_THREADS)))


def limit_cpu(n: int | None = None, *, quiet: bool = False) -> dict:
    """Cap every CPU pool this stack can reach. Returns what was achieved.

    Safe to call more than once and safe to call after torch is imported —
    `torch.set_num_threads` is a runtime setting. The env vars are set too, for
    any library loaded later in the process, but a library already loaded keeps
    the pool it sized at import; that is what the launcher's export is for.
    """
    n = requested_threads(n)
    for var in _ENV_VARS:
        os.environ[var] = str(n)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    got: dict = {"requested": n}
    try:
        import torch
        torch.set_num_threads(n)
        # Interop is a SEPARATE pool and can only be set before the first
        # parallel region; failing is normal and not worth an exception.
        try:
            torch.set_num_interop_threads(n)
        except RuntimeError:
            pass
        got["torch"] = torch.get_num_threads()
        got["torch_interop"] = torch.get_num_interop_threads()
    except ImportError:
        pass
    try:
        import pyarrow as pa
        pa.set_cpu_count(n)
        pa.set_io_thread_count(max(2, n))
        got["pyarrow"] = pa.cpu_count()
        got["pyarrow_io"] = pa.io_thread_count()
    except (ImportError, AttributeError):
        pass
    try:
        # fev binds DEFAULT_NUM_PROC = multiprocessing.cpu_count() as the
        # default of iter_windows / load_full_dataset / get_window; rebinding
        # the constant caps every fev path uniformly.
        import fev.constants
        fev.constants.DEFAULT_NUM_PROC = n
        got["fev"] = fev.constants.DEFAULT_NUM_PROC
    except ImportError:
        pass

    over = {k: v for k, v in got.items() if k != "requested" and v > n}
    if over and not quiet:
        logger.warning(
            "CPU cap %d requested but %s — a pool was sized before this ran; "
            "export the env vars before starting python", n, over)
    elif not quiet:
        logger.info("CPU cap: %d thread(s) — %s", n,
                    ", ".join(f"{k}={v}" for k, v in got.items()
                              if k != "requested"))
    return got
