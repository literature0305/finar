#!/usr/bin/env python3
"""Score one model twice: a plain multivariate pass, then a pseudo-known-future pass.

    step 1   multivariate forecast of every variate
    step 2   variate 0 again, with variates 1..n-1 handed back as
             past_covariates + future_covariates, the future filled from step 1

Only variate 0 is scored, in both steps, so the pair answers one question about
one series. See `two_step.py` for why the task has to be restructured and
`datasets_iter.py` for why the benchmarks are read directly rather than through
their adapters.

The question is whether a model trained WITHOUT iterative refinement gains from
its own covariate forecasts presented as observations. Chronos-2 and TiRex-2
are one-pass by construction; EO v4 is made one-pass with `--coe-eval-depth 1`.

STEP 2 IS ONLY MEANINGFUL IF IT DIFFERS FROM STEP 1. `Chronos2Forecaster` and
`EOForecaster` resolve `supports_covariates` through the same function as
`supports_multivariate`, so both steps can reach the model by one path. Every
run records how many tasks came back identical and refuses one where all of
them did — that failure looks exactly like "no gain" otherwise.

tsm-trainer is imported, never written to.

Usage
-----
    python run_eval.py --model-path amazon/chronos-2 --out <dir>
    python run_eval.py --model-path google/timesfm-3.0-pytorch ...
    python run_eval.py --model-path NX-AI/TiRex-2 ...
    python run_eval.py --model-path <eo-v4 ckpt> --coe-eval-depth 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

logger = logging.getLogger("finar_exp005")

DEFAULT_REPO = Path("/group-volume/workspace/mun-hak.lee/experiments/"
                    "tsm-trainer_001/tsm-trainer")
_LOCAL_FEV = Path("/group-volume/ts-dataset/benchmarks/fev_bench")
_LOCAL_GIFT = Path("/group-volume/ts-dataset/benchmarks/gift_eval")
BENCHMARKS = ("fev", "gift")

#: The quantile grid every model is asked for. 0.5 must be present: the point
#: forecast MASE scores, and the covariate future step 2 is handed, are both
#: the median.
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def add_repo_to_path(repo: Path) -> None:
    ev = repo / "scripts" / "forecasting" / "evaluation"
    if not ev.is_dir():
        raise SystemExit(f"no evaluation package under {repo} (looked at {ev})")
    for p in (str(ev), str(repo / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def resolve_data(explicit, default: Path, what: str):
    if explicit:
        return str(explicit)
    if default.is_dir():
        logger.info("%s data: %s (local, offline)", what, default)
        return str(default)
    logger.warning("%s data: no local cache at %s — the Hub path needs network",
                   what, default)
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True,
                   help="amazon/chronos-2 | google/timesfm-3.0-pytorch | "
                        "NX-AI/TiRex-2 | a local EO v4 checkpoint")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS),
                   choices=list(BENCHMARKS))
    p.add_argument("--coe-eval-depth", type=int, default=1,
                   help="EO v4 only. 1 makes a K>=2 checkpoint behave as the "
                        "one-pass model this experiment is about; ignored by "
                        "the other three.")
    p.add_argument("--alpha", type=float, nargs="+", default=[0.0],
                   help="how much of the covariates' TRUE future to blend into "
                        "the pseudo one, each in [0, 1]: "
                        "alpha*oracle + (1-alpha)*pseudo. 0 is the scenario "
                        "exp005 measured (the model's own guess); 1 is the "
                        "known-future upper bound. Only COVARIATES are ever "
                        "oracle — the scored target never is.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--max-items-per-task", type=int, default=None)
    p.add_argument("--max-windows", type=int, default=8,
                   help="fev only: windows per task. Its multivariate tasks "
                        "carry very few series, so one window is a two-item "
                        "task. GIFT-Eval's items already are windows x series.")
    p.add_argument("--task-subset-index", type=int, default=0)
    p.add_argument("--num-task-subsets", type=int, default=1)
    p.add_argument("--fev-data", default=None)
    p.add_argument("--gift-data", default=None)
    return p.parse_args(argv)


def build_forecaster(args):
    """The model, refused up front if step 2 could never differ from step 1."""
    from run_benchmark import load_forecaster

    forecaster = load_forecaster(SimpleNamespace(
        model_path=args.model_path, device="cuda", torch_dtype="float32",
        config_path=None, model_source_path=None, batch_size=args.batch_size,
        eo_max_n_variate=None, rolling_horizon=None, rolling_quantiles=None,
        coe_eval_depth=args.coe_eval_depth))
    if not getattr(forecaster, "supports_covariates", False):
        raise SystemExit(
            f"{args.model_path} reports supports_covariates=False, so step 2's "
            f"future_covariates would never reach it and both steps would "
            f"return the same forecast.")
    logger.info("model %s: multivariate=%s covariates=%s", args.model_path,
                getattr(forecaster, "supports_multivariate", None),
                getattr(forecaster, "supports_covariates", None))
    return forecaster


def score_task(forecaster, task, group, args, Metrics, two_step) -> list[dict]:
    """One row per alpha for this task, all sharing ONE step 1 and ONE mask."""
    H = int(group[0]["horizon"])
    # Before the forward. An item carrying an all-NaN variate makes the model
    # return fewer rows than it was given, unlabelled, so variate 0 can no
    # longer be identified — see `two_step.forecastable`. 3 of 600 items in
    # bitbrains_fast_storage/H/short are like this.
    ok = two_step.forecastable(group)
    n_unforecastable = int((~ok).sum())
    if n_unforecastable:
        logger.info("  %s: dropping %d/%d items with an all-NaN variate",
                    task, n_unforecastable, len(group))
        group = [it for it, k in zip(group, ok) if k]
    if not group:
        return []

    season = Metrics.get_seasonal_period(group[0]["freq"])
    t0 = time.time()
    # ONE step 1 for every alpha: it does not depend on the blend, so paying it
    # per alpha would cost a forward each and let the sweep's own baseline drift
    # between its columns.
    with torch.no_grad():
        s1, cov_future = two_step.run_step1(
            forecaster, group, H, QUANTILES, batch_size=args.batch_size)
    # ONE prepare for every step and every alpha: scoring each on its own
    # survivors would compare different populations.
    prep = two_step.prepare(group, H, season, Metrics)
    a = two_step.score(s1, prep, QUANTILES, Metrics)

    rows = []
    for alpha in args.alpha:
        with torch.no_grad():
            s2, ident = two_step.run_step2(
                forecaster, group, cov_future, s1, alpha, H, QUANTILES,
                batch_size=args.batch_size)
        b = two_step.score(s2, prep, QUANTILES, Metrics)
        imp = ((a["MASE"] - b["MASE"]) / a["MASE"] * 100
               if np.isfinite(a["MASE"]) and a["MASE"] else float("nan"))
        rows.append({
            "model": str(args.model_path), "task": task, "alpha": alpha,
            "n_items": len(group), "horizon": H, "seasonality": season,
            "n_variates": int(group[0]["context"].shape[0]),
            "n_scored": prep["n_scored"], "n_dropped": prep["n_dropped"],
            "n_unforecastable": n_unforecastable,
            "step1_MASE": a["MASE"], "step2_MASE": b["MASE"],
            "step1_WQL": a["WQL"], "step2_WQL": b["WQL"],
            "MASE_improvement_pct": imp,
            "win_rate": two_step.win_rate(a, b), "n_identical": ident,
            "elapsed_s": round(time.time() - t0, 1),
        })
    return rows


def run_benchmark(forecaster, benchmark: str, args, Metrics, two_step,
                  load_items, totals) -> list[dict]:
    root = (resolve_data(args.fev_data, _LOCAL_FEV, "fev") if benchmark == "fev"
            else resolve_data(args.gift_data, _LOCAL_GIFT, "gift"))
    items = list(load_items(
        benchmark, data_dir=root, max_tasks=args.max_tasks,
        max_items_per_task=args.max_items_per_task,
        max_windows=args.max_windows,
        task_subset=(args.task_subset_index, args.num_task_subsets)))
    if not items:
        logger.warning("%s: no multivariate items", benchmark)
        return []
    # Grouped by task, because a horizon and a seasonality are properties of
    # the task and a batch has to share both.
    by_task: dict[str, list] = {}
    for it in items:
        by_task.setdefault(it["task"], []).append(it)
    del items                   # by_task holds every one of them already
    logger.info("%s: %d task(s), %d item(s)", benchmark, len(by_task),
                sum(len(g) for g in by_task.values()))

    rows = []
    for task in list(by_task):
        group = by_task.pop(task)   # freed once this task is scored
        task_rows = score_task(forecaster, task, group, args, Metrics, two_step)
        if not task_rows:
            logger.warning("%s: every item carries an all-NaN variate — skipped",
                           task)
            continue
        for row in task_rows:
            row["benchmark"] = benchmark
            rows.append(row)
            totals["identical"] += row["n_identical"]
            totals["items"] += row["n_items"]
            if row["n_identical"] == row["n_items"]:
                # The whole-run refusal below cannot see this: a model blind on
                # SOME tasks still differs on others, so the run passes while
                # these rows carry a null that is plumbing, not a finding.
                # Toto-2 is the known case — it reads only the first
                # ceil(H/patch)-1 patches of a known future, so every task with
                # H <= 32 gets its future dropped entirely.
                logger.warning(
                    "  %s (alpha=%g): step 2 returned step 1 on ALL %d items "
                    "(H=%d). This row measures nothing — the covariate future "
                    "did not reach the model.", task, row["alpha"],
                    row["n_items"], row["horizon"])
            logger.info("  %-30s a=%-4.2f n=%-4d(-%d) MASE %.4f -> %.4f "
                        "(%+.2f%%)  win %.0f%%  identical %d/%d",
                        task[:30], row["alpha"], row["n_scored"],
                        row["n_dropped"], row["step1_MASE"], row["step2_MASE"],
                        row["MASE_improvement_pct"], row["win_rate"] * 100,
                        row["n_identical"], row["n_items"])
    return rows


def main() -> int:
    args = parse_args()
    bad = [a for a in args.alpha if not 0.0 <= a <= 1.0]
    if bad:
        raise SystemExit(f"--alpha must be in [0, 1]; got {bad}. It is the "
                         f"share of the covariates' true future in a convex "
                         f"blend with the model's own forecast of them.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    add_repo_to_path(args.repo)

    from engine.metrics import MetricRegistry as Metrics  # noqa: E402

    import two_step                                       # noqa: E402
    from datasets_iter import load_items                  # noqa: E402

    args.out.mkdir(parents=True, exist_ok=True)
    # Sharded runs must not collide: without the suffix, `--task-subset-index 1`
    # overwrites shard 0's csv in the same --out and the merge silently loses
    # half the tasks (see `base.py::_split_suffix`).
    model_tag = Path(str(args.model_path)).name + (
        f"_shard{args.task_subset_index}of{args.num_task_subsets}"
        if args.num_task_subsets > 1 else "")
    forecaster = build_forecaster(args)

    rows, totals = [], {"identical": 0, "items": 0}
    for benchmark in args.benchmarks:
        rows += run_benchmark(forecaster, benchmark, args, Metrics, two_step,
                              load_items, totals)

    if not rows:
        raise SystemExit("no rows scored — check --benchmarks and the data roots")
    if totals["identical"] == totals["items"]:
        raise SystemExit(
            f"step 2 returned step 1 on ALL {totals['items']} items. The "
            f"pseudo-known-future covariates never changed the forecast, so "
            f"this run measures nothing — it is not a null result. Check that "
            f"the model consumes future_covariates.")
    import pandas as pd
    dest = args.out / f"{model_tag}.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    (args.out / f"{model_tag}.meta.json").write_text(json.dumps({
        "model_path": str(args.model_path), "quantiles": QUANTILES,
        "alpha": list(args.alpha),
        "coe_eval_depth": args.coe_eval_depth,
        "identical_items": totals["identical"], "total_items": totals["items"],
        "supports_multivariate": bool(
            getattr(forecaster, "supports_multivariate", False)),
    }, indent=1))
    logger.info("wrote %s (%d rows, %d/%d items identical)", dest, len(rows),
                totals["identical"], totals["items"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
