#!/usr/bin/env python3
"""One CPU cap for every finar experiment — derived from the allocation, not typed.

WHY THIS EXISTS
---------------
`run_benchmark.py` caps threads in two places: `_apply_cpu_thread_limit` (its
line 87), which sets the env-var pools and is importable, and a block under
`if __name__ == "__main__":` (its 2606-2620) doing `torch.set_num_threads`,
`pa.set_cpu_count` and `pa.set_io_thread_count` — which is NOT importable, since
that file has no `main()`. Every finar experiment imports `load_forecaster` and
drives the adapters itself, so it got the first and not the second. Measured on
an 18-core box: torch 18, torch interop 18, fev's bound `num_proc` default 18,
against an intended 8 — 17.8 effective cores where 8 were asked for.

THE CAP IS READ, NOT TYPED
--------------------------
The incident this exists for is a node that reports 255 logical CPUs while
allocating ~29, because `multiprocessing.cpu_count()` ignores cgroup quotas.
Asking an operator to know the allocation and retype it as `--num-workers 29` in
seven places restates the quota as folklore, and gets it wrong on the next node.
`available_cpus()` reads it: the scheduler affinity mask, the cgroup v2/v1 CPU
quota, and `cpu_count()`, whichever is smallest.

An explicit request is CLAMPED to that, never trusted above it. Without the
clamp the helper inverts its own purpose: on a node where something already
exported `OMP_NUM_THREADS=255`, the env fallback would return 255 and this
module would then write 255 into every other pool — becoming the amplifier
rather than the cap.

ORDER MATTERS, AND HALF OF IT IS NOT RUNTIME-SETTABLE
-----------------------------------------------------
OpenMP and BLAS size their pools when the library is first loaded, so
`OMP_NUM_THREADS` must be in the environment before `import numpy` or
`import torch`. That is why every entry point calls `limit_cpu()` at MODULE
scope, above its own third-party imports, and again from `main()` once
`--num-workers` is parsed — the second call is runtime-only work (torch,
pyarrow, fev) and cannot be undone by import order.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger("finar_cpu")

#: Ceiling for the unquota'd case, where the affinity mask is the whole machine
#: and there is nothing else to go on. The failure mode of too few threads is a
#: slow run; of too many, a killed one.
DEFAULT_MAX = 8

#: The same six names `run_benchmark._apply_cpu_thread_limit` sets (its line 87)
#: — copied because that function uses `setdefault`, so it cannot LOWER a value
#: already in the environment, which is exactly the case here. If a seventh is
#: added upstream it must be added here too.
_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_MAX_THREADS", "TOKIO_WORKER_THREADS",
             "HF_XET_NUM_CONCURRENT_RANGE_GETS")


def _cgroup_quota() -> int | None:
    """CPUs this cgroup may use, or None when unquota'd or unreadable."""
    try:                                        # cgroup v2
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:                                        # cgroup v1
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0 and p > 0:
            return max(1, int(q / p))
    except (OSError, ValueError):
        pass
    return None


def available_cpus() -> int:
    """How many CPUs this process may actually use.

    The smallest of: the scheduler affinity mask, the cgroup quota, and
    `cpu_count()`. `cpu_count()` alone is what reports 255 on a node that
    allocates 29.
    """
    candidates = [os.cpu_count() or 1]
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    q = _cgroup_quota()
    if q:
        candidates.append(q)
    return max(1, min(candidates))


def requested_threads(explicit: int | None = None) -> int:
    """The cap to use: the flag, else the environment, else the default —
    always clamped to what the allocation actually permits."""
    allowed = available_cpus()
    if explicit:
        want = int(explicit)
    else:
        try:
            want = int(os.environ.get("OMP_NUM_THREADS", "") or 0)
        except ValueError:
            want = 0
        want = want or min(DEFAULT_MAX, allowed)
    n = max(1, min(want, allowed))
    if explicit and n != int(explicit):
        logger.warning("--num-workers %s exceeds the %d CPU(s) this process is "
                       "allocated; using %d", explicit, allowed, n)
    return n


def argv_workers(argv=None) -> int | None:
    """`--num-workers N` read straight off the command line, or None.

    The module-scope call happens before argparse exists, and torch's INTER-op
    pool can only be sized once — so without this the early call would fix it at
    the derived default and `--num-workers 4` could never lower it. Deliberately
    forgiving: an unparsable value falls through to the derived default rather
    than failing here, because argparse will report it properly moments later.
    """
    argv = sys.argv if argv is None else argv
    for i, a in enumerate(argv):
        if a == "--num-workers" and i + 1 < len(argv):
            v = argv[i + 1]
        elif a.startswith("--num-workers="):
            v = a.split("=", 1)[1]
        else:
            continue
        try:
            return int(v)
        except ValueError:
            return None
    return None


def limit_cpu(n: int | None = None, *, quiet: bool = False) -> int:
    """Cap every CPU pool this stack can reach. Returns the cap applied.

    Idempotent. Call once at module scope, above numpy/torch, so the env-var
    pools are sized correctly, and again once the flag is parsed.
    """
    n = requested_threads(n)
    for var in _ENV_VARS:
        os.environ[var] = str(n)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    got: dict = {}
    try:
        import torch
        torch.set_num_threads(n)
        try:
            # A separate pool, settable only before the first parallel region.
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
    # ONLY if fev is already loaded. `_cap_fev_num_proc` imports fev, which
    # pulls the whole datasets/HF stack: measured 0.54 s and +88 MB RSS, paid by
    # the three experiments that never touch fev. The four that do have it
    # loaded by the time this runs from main(), and cap it to 1 themselves
    # downstream anyway.
    if "fev" in sys.modules:
        try:
            from benchmarks.fev_bench import _cap_fev_num_proc
            _cap_fev_num_proc(n)
            import inspect

            import fev
            # Read back the BOUND DEFAULT, not the constant: fev binds
            # DEFAULT_NUM_PROC as a default argument, and a function's defaults
            # are captured at definition time — setting the constant leaves
            # `iter_windows(num_proc=...)` at the machine core count while
            # reporting success.
            got["fev"] = int(inspect.signature(
                fev.Task.iter_windows).parameters["num_proc"].default)
        except Exception:  # noqa: BLE001 — fev absent or its signature moved
            pass

    over = {k: v for k, v in got.items()
            if v > (max(2, n) if k == "pyarrow_io" else n)}
    if over and not quiet:
        logger.warning(
            "CPU cap %d requested but %s — a pool was sized before this ran. "
            "Call limit_cpu() at module scope, above numpy/torch.", n, over)
    elif not quiet:
        logger.info("CPU cap: %d of %d allocated — %s", n, available_cpus(),
                    ", ".join(f"{k}={v}" for k, v in got.items()) or "no pools yet")
    return n

