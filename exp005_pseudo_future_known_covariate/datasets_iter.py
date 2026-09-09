#!/usr/bin/env python3
"""The multivariate items of fev-bench and GIFT-Eval, as plain arrays.

Both benchmarks are read through their own libraries rather than reimplemented,
but neither adapter is used to SCORE: exp005 restructures the task between its
two steps and scores variate 0 only, which no adapter's aggregate can express
(step 1 scores n targets, step 2 scores one).

An item is ``{"context": (V, T), "truth": (V, H), "freq": str, "task": str}``.
Variate 0 is the designated target; it is variate 0 of the stored array, so it
is the same series in both steps and in every model's run — the same positional
guarantee `exp002_feedback_factorization_pseudo-target` relies on.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("finar_exp005")

#: fev-bench's own name for ">1 target column and no dynamic covariates".
FEV_SUBSET = "multivariate"


def fev_items(data_dir, max_tasks=None, max_items_per_task=None,
              task_subset=(0, 1), max_windows=8):
    """Yield the multivariate fev-bench windows.

    The truth comes from the window's `test_data`, not from `get_input_data()`'s
    future half: fev strips the target columns from that half because they are
    the answer.
    """
    from benchmarks.fev_bench import (FevBenchAdapter, _cap_fev_num_proc,
                                      _localize_task)

    # The ADAPTER's own loader, not a second reading of the yaml: it owns the
    # vendored task list, the subset filter and the fev-install check, and a
    # private copy here would be the thing that goes stale when the task list
    # is re-vendored. The public `load_tasks()` is NOT the alternative — it
    # returns metadata dicts and discards the fev.Task objects, so
    # `iter_windows()` is unreachable through it. `evaluate()` itself does
    # exactly these two lines.
    i, n = task_subset
    adapter = FevBenchAdapter(data_dir=data_dir, subset=FEV_SUBSET,
                              task_subset_index=i, num_task_subsets=n)
    benchmark = adapter._load_benchmark()
    # The adapter's own sharding, not a hand-rolled slice: index slicing only
    # tiles the task set if every shard sorts the list the same way first, and
    # `_apply_task_subset` is where that canonical sort lives. It returns the
    # list unchanged when n == 1.
    tasks = adapter._apply_task_subset(
        [t for t in benchmark.tasks if adapter._filter_task(t)])
    if max_tasks:
        tasks = tasks[:max_tasks]
    logger.info("fev: %d multivariate task(s)", len(tasks))

    # fev binds cpu_count() as its default num_proc, which ignores cgroup
    # limits and forks hundreds of workers that thrash during window loading.
    _cap_fev_num_proc(1)

    for task in tasks:
        # The IDENTITY is read before localizing: _localize_task clears
        # dataset_config (that is how it forces fev's local-file code path), so
        # the name has to come off the original task or every row is "None".
        name, H = str(task.dataset_config), int(task.horizon)
        cols = list(task.target_columns)
        # fev's loader takes a Hub code path whenever dataset_config is set,
        # even with the parquet already on disk. Without this the "local,
        # offline" the caller logs is not true.
        task, _is_local = _localize_task(task, data_dir)
        # `task.freq` is a property that RAISES until the dataset is loaded, and
        # iter_windows() is what loads it — so it cannot be hoisted out of the
        # loop, only cached after the first window.
        freq = None
        # SEVERAL windows, not one. fev's multivariate tasks carry very few
        # SERIES — ETT is one multivariate series per window — so a single
        # window gives a two-item task and no statistic worth reading. The
        # windows of a task share its horizon and seasonality, so they pool.
        for w, window in enumerate(task.iter_windows()):
            if w >= max_windows:
                break
            # ONE pass over the Arrow data. get_input_data() and
            # get_ground_truth() each call this uncached, re-running the column
            # select and the past/future split to throw half the result away.
            past, _future_known, gt = window._get_past_future_test_data()
            if freq is None:
                freq = str(task.freq)
            by_col_past = {c: past[c] for c in cols}
            by_col_gt = {c: np.asarray(gt[c], dtype=np.float32) for c in cols}
            m = len(past) if max_items_per_task is None else min(
                len(past), max_items_per_task)
            for k in range(m):
                yield {
                    "context": np.stack(
                        [np.asarray(by_col_past[c][k], dtype=np.float32)
                         for c in cols], axis=0),
                    "truth": np.stack([by_col_gt[c][k] for c in cols], axis=0),
                    "freq": freq, "task": name, "horizon": H,
                }


def gift_multivariate_names() -> set[str]:
    """GIFT-Eval dataset names whose targets are multivariate.

    From tsm-trainer's committed `baselines/gift_eval_hf_variate_types.csv`,
    which is keyed by the same ``base/freq`` names this module enumerates and
    carries `target_dim`. A hand-kept tuple here would go stale silently the
    next time GIFT-Eval gains a multivariate dataset, and a wrong entry yields
    univariate items where this experiment is undefined — there would be no
    covariate to predict.

    The CSV is read, not `get_gift_eval_hf_variate_types()`: that function
    WRITES the file back when it meets an unknown name, and the tsm-trainer
    checkout is read-only here.
    """
    import pandas as pd
    from benchmarks import variate_types

    df = pd.read_csv(variate_types._GIFT_EVAL_HF_CSV)
    return set(df.loc[df["is_multivariate"].astype(bool), "dataset"])


def gift_items(data_dir, max_tasks=None, max_items_per_task=None,
               task_subset=(0, 1), term="short"):
    """Yield the multivariate GIFT-Eval windows."""
    import os

    if data_dir:
        os.environ["GIFT_EVAL"] = str(data_dir)
    from gift_eval.data import Dataset

    multivariate = gift_multivariate_names()
    root = os.environ.get("GIFT_EVAL", "")
    names = []
    for base in sorted(os.listdir(root) if os.path.isdir(root) else []):
        d = os.path.join(root, base)
        if not os.path.isdir(d):
            continue
        subs = [f"{base}/{s}" for s in sorted(os.listdir(d))
                if os.path.isdir(os.path.join(d, s))] or [base]
        names += [s for s in subs if s in multivariate]
    # Sorted above, so every shard slices the same order (see
    # `base.py::_apply_task_subset` for what happens when they do not).
    i, n = task_subset
    names = names[i * len(names) // n:(i + 1) * len(names) // n]
    if max_tasks:
        names = names[:max_tasks]
    logger.info("gift: %d multivariate dataset(s)", len(names))

    for name in names:
        try:
            ds = Dataset(name=name, term=term, to_univariate=False)
        except Exception as e:                       # noqa: BLE001
            logger.warning("skipping %s (%s: %s)", name, type(e).__name__, e)
            continue
        if int(getattr(ds, "target_dim", 1)) < 2:
            logger.warning("skipping %s — target_dim < 2, so there is no "
                           "covariate to predict", name)
            continue
        H, freq, task = int(ds.prediction_length), str(ds.freq), f"{name}/{term}"
        for k, (inp, lbl) in enumerate(ds.test_data):
            if max_items_per_task is not None and k >= max_items_per_task:
                break
            yield {
                "context": np.atleast_2d(
                    np.asarray(inp["target"], dtype=np.float32)),
                "truth": np.atleast_2d(
                    np.asarray(lbl["target"], dtype=np.float32))[:, :H],
                "freq": freq, "task": task, "horizon": H,
            }


def load_items(benchmark: str, **kw):
    if benchmark.startswith("fev"):
        return fev_items(**kw)
    if benchmark.startswith("gift"):
        # GIFT-Eval has no per-task window count to cap: its items already ARE
        # windows x series. Accepting the argument and ignoring it is what made
        # --max-windows silently a no-op on half the run.
        kw.pop("max_windows", None)
        return gift_items(**kw)
    raise ValueError(f"unknown benchmark {benchmark!r}")
