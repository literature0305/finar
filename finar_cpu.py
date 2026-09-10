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
from pathlib import Path

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
    # Reported, not silently reconciled: OpenMP and BLAS sized their pools when
    # the library loaded, so a cap that disagrees with what the environment said
    # at startup is half-applied whatever we write now. The launcher passes the
    # same number to both, so this only fires on a hand-rolled invocation.
    env_n = os.environ.get("OMP_NUM_THREADS")
    if env_n and int(env_n) != n and not quiet:
        logger.warning(
            "CPU cap %d was requested but the process started with "
            "OMP_NUM_THREADS=%s; the OpenMP/BLAS pools are already that size "
            "and cannot be resized from here. Pass one value to both, or set "
            "the environment before starting python.", n, env_n)
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
    # fev binds DEFAULT_NUM_PROC = multiprocessing.cpu_count() as the DEFAULT
    # ARGUMENT of iter_windows / load_full_dataset / get_window. Rebinding the
    # constant does NOT change those: a function's defaults are captured in
    # __defaults__ at definition time. Verified — after setting the constant to
    # 8, `signature(fev.Task.iter_windows).parameters['num_proc'].default` is
    # still 18. tsm-trainer's `_cap_fev_num_proc` rebinds the bound defaults,
    # which is exactly why it exists; reporting the constant instead would be a
    # false success while fev still forks one worker per machine core.
    try:
        from benchmarks.fev_bench import _cap_fev_num_proc
        _cap_fev_num_proc(n)
    except ImportError:
        pass
    try:
        import inspect

        import fev
        got["fev"] = int(inspect.signature(
            fev.Task.iter_windows).parameters["num_proc"].default)
    except Exception:  # noqa: BLE001 — fev absent or its signature changed
        pass

    # `max(2, n)` is what pyarrow's io pool was asked for, so compare against
    # that: at n = 1 a correctly applied cap would otherwise always warn.
    ceilings = {"pyarrow_io": max(2, n)}
    over = {k: v for k, v in got.items()
            if k != "requested" and v > ceilings.get(k, n)}
    if over and not quiet:
        logger.warning(
            "CPU cap %d requested but %s — a pool was sized before this ran; "
            "export the env vars before starting python", n, over)
    elif not quiet:
        logger.info("CPU cap: %d thread(s) — %s", n,
                    ", ".join(f"{k}={v}" for k, v in got.items()
                              if k != "requested"))
    return got


def load_from(start: Path):
    """This module, found from a caller's own location, without touching sys.path.

    Each experiment directory is meant to be self-contained enough to copy to a
    remote host, and inserting the repo root on `sys.path` to import this would
    also expose every sibling name to shadowing. Callers use::

        from finar_cpu import load_from        # if the root is importable
        limit_cpu = load_from(Path(__file__)).limit_cpu

    but the supported form is `bootstrap()` below, which needs no import at all.
    """
    return _bootstrap(start)


def _bootstrap(start: Path):
    import importlib.util
    for d in [start.resolve().parent, *start.resolve().parents]:
        f = d / "finar_cpu.py"
        if f.is_file():
            spec = importlib.util.spec_from_file_location("finar_cpu", f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit(
        f"finar_cpu.py not found above {start} — it holds the CPU cap every "
        f"experiment needs, and without it a run sizes its thread pools to the "
        f"whole machine. Copy it next to the experiment directory.")
