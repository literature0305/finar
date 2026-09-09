#!/usr/bin/env python3
"""Score one EO v4 checkpoint with the TRUE horizon fed into its second pass.

Runs the same checkpoint at several values of alpha in

    intermediate_pred_new = (1 - alpha) * intermediate_pred + alpha * target

so the trend from "the model's own forecast" (alpha = 0) to "the ground truth"
(alpha = 1) can be read off. See `truth_patch.py` for where the truth comes
from and where the blend is applied.

ALPHA > 0 IS LABEL LEAKAGE BY CONSTRUCTION. These numbers are a diagnostic
upper bound on what a second pass can do with a better input; they are not a
benchmark score and must not be quoted as one. alpha = 0 is the stock run and
is the only row here comparable to anything else.

Iteration 1 does not depend on alpha — the blend happens between pass 1 and
pass 2 — so `build_table.py` asserts that `repeat1_*` agrees across the sweep.
A disagreement means the patch reached pass 1.

tsm-trainer is imported, never written to.

Usage
-----
    python run_eval.py --model-path <ckpt> --repo <tsm-trainer> --out <dir>
    python run_eval.py ... --alpha 0.0 0.5 1.0      # the default sweep
    python run_eval.py ... --benchmarks fev         # one benchmark
    python run_eval.py ... --task-subset-index 0 --num-task-subsets 4
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

logger = logging.getLogger("finar_exp004")

DEFAULT_REPO = Path("/group-volume/workspace/mun-hak.lee/experiments/"
                    "tsm-trainer_001/tsm-trainer")
_LOCAL_FEV = Path("/group-volume/ts-dataset/benchmarks/fev_bench")
_LOCAL_GIFT = Path("/group-volume/ts-dataset/benchmarks/gift_eval")

BENCHMARKS = ("fev", "gift")
DEFAULT_ALPHAS = (0.0, 0.5, 1.0)


def add_repo_to_path(repo: Path) -> None:
    ev = repo / "scripts" / "forecasting" / "evaluation"
    if not ev.is_dir():
        raise SystemExit(f"no evaluation package under {repo} (looked at {ev})")
    for p in (str(ev), str(repo / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def warmup_depth(eo: dict) -> int:
    """How deep the grad-free warm-up runs before the scored chain.

    `init_from_repeat_prediction` is only the SWITCH; the depth lives in
    `init_warmup_depth_max`, and its default is the trap — null follows
    `coe_train_depth_max` and is INERT at 1, so the switch reads true while
    nothing happens. `coe_bottleneck` is part of the gate too: without it the
    passes chain in hidden space and the warm-up never runs. This mirrors
    tsm-trainer's `AEDForecastingConfig.warmup_active`, which folds all three
    into one place precisely because writing them out twice had already let
    the model and the config disagree.
    """
    if not (eo.get("init_from_repeat_prediction") and eo.get("coe_bottleneck")):
        return 0
    d = eo.get("init_warmup_depth_max")
    if d is None:
        k = eo.get("coe_train_depth_max")
        k = k if isinstance(k, int) else 1
        return 0 if k <= 1 else k
    return int(d) if isinstance(d, int) and d > 0 else 0


#: Why each field is required, kept beside the check rather than interleaved
#: with the control flow that raises.
_WHY_REQUIRED = {
    "coe_residual":
        "Pass 2 then predicts an absolute forecast rather than a residual "
        "against the accumulator, so there is nothing to keep consistent with "
        "the blended input.",
    "coe_bottleneck":
        "Without the bottleneck the passes chain in hidden space and never "
        "write a forecast back, so there is no intermediate prediction to "
        "replace.",
}


def require_coe(model_path: str) -> dict:
    """Refuse a checkpoint this experiment is not defined for.

    `coe_residual` is the one this experiment adds to the usual pair: pass 2
    predicts a residual against an accumulator seeded from what it was handed,
    and the whole design rests on that accumulator being re-derived from the
    BLENDED median. Without it there is no accumulator and no residual to keep
    consistent, so the work order's warning has nothing to attach to.
    """
    cfg = Path(model_path) / "config.json"
    if not cfg.is_file():
        raise SystemExit(f"{cfg} not found — this needs a local EO v4 checkpoint")
    eo = (json.loads(cfg.read_text()).get("eo_config") or {})
    if not eo:
        raise SystemExit(f"{model_path} has no eo_config — not an EO v4 model")
    for field, why in _WHY_REQUIRED.items():
        if eo.get(field) is not True:
            raise SystemExit(f"{field}={eo.get(field)!r}, not True. {why}")
    # The TRAINED depth, not `coe_train_depth_max` alone: a checkpoint at K=1
    # with a grad-free warm-up of 1 has been trained on an input a pass already
    # refined, and has a second pass to feed. Testing K alone refuses it —
    # which is what this file did until the other four experiments' guard was
    # compared against it.
    k = eo.get("coe_train_depth_max")
    k = k if isinstance(k, int) and k >= 1 else 1
    warm = warmup_depth(eo)
    if k + warm < 2:
        raise SystemExit(
            f"trained depth {k + warm} < 2 (coe_train_depth_max={k}, grad-free "
            f"warm-up {warm}): this checkpoint has no second pass to feed "
            f"anything into.\n  init_from_repeat_prediction="
            f"{eo.get('init_from_repeat_prediction')!r} "
            f"init_warmup_depth_max={eo.get('init_warmup_depth_max')!r}")
    logger.info("checkpoint: trained_depth=%s (K=%s + warm-up %s) residual=%s "
                "bottleneck=%s repeat_encoding=%s", k + warm, k, warm,
                eo.get("coe_residual"), eo.get("coe_bottleneck"),
                eo.get("coe_repeat_encoding"))
    return eo


def resolve_data(explicit, default: Path, what: str):
    if explicit:
        return str(explicit)
    if default.is_dir():
        logger.info("%s data: %s (local, offline)", what, default)
        return str(default)
    logger.warning("%s data: no local cache at %s — falling back to the Hub",
                   what, default)
    return None


def make_adapter(benchmark: str, args):
    """The stock adapter, with variates fed as one group.

    `ignore_group_id=False` in both: it is what feeds a task's variates
    together instead of flattening them into independent univariate series.
    Every variate then has a row the write-back covers, which is what "the
    truth is fed back for every variate" means operationally.
    """
    slice_kw = {"task_subset_index": args.task_subset_index,
                "num_task_subsets": args.num_task_subsets}
    if benchmark == "fev":
        from benchmarks.fev_bench import FevBenchAdapter
        return FevBenchAdapter(
            data_dir=resolve_data(args.fev_data, _LOCAL_FEV, "fev"),
            subset=args.fev_subset, batch_size=args.batch_size,
            ignore_group_id=False, **slice_kw), "fev_bench"
    if benchmark == "gift":
        from benchmarks.gift_eval_hf_mul import GiftEvalHFMulAdapter
        return GiftEvalHFMulAdapter(
            data_dir=resolve_data(args.gift_data, _LOCAL_GIFT, "gift"),
            batch_size=args.batch_size, ignore_group_id=False,
            name="gift_eval_hf_mul", **slice_kw), "gift_eval_hf_mul"
    raise SystemExit(f"unknown benchmark {benchmark!r}; known: {BENCHMARKS}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--alpha", type=float, nargs="+", default=list(DEFAULT_ALPHAS),
                   help="blend weights to sweep (default: 0.0 0.5 1.0)")
    p.add_argument("--coe-eval-depth", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    p.add_argument("--fev-subset", default="multivariate",
                   choices=("all", "univariate", "multivariate", "covariate"),
                   help="fev-bench subset. Default `multivariate`: every "
                        "variate is then a target, so 'the truth is fed back "
                        "for every variate' has no covariate exception")
    p.add_argument("--fev-data", default=None)
    p.add_argument("--gift-data", default=None)
    p.add_argument("--task-subset-index", type=int, default=0)
    p.add_argument("--num-task-subsets", type=int, default=1)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.coe_eval_depth < 2:
        raise SystemExit("--coe-eval-depth must be >= 2: this is an "
                         "iteration-1 vs iteration-2 comparison")
    bad = [b for b in args.benchmarks if b not in BENCHMARKS]
    if bad:
        raise SystemExit(f"unknown benchmark(s) {bad}; known: {list(BENCHMARKS)}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    add_repo_to_path(args.repo)
    from truth_patch import (TruthStore, capture_truth,  # noqa
                             require_repo_support, truth_feedback)

    # Before the checkpoint is loaded, not from inside the patch mid-sweep.
    require_repo_support()
    eo = require_coe(args.model_path)

    from run_benchmark import load_forecaster  # noqa: E402

    # ASSIGNED, not setdefault: an inherited TSM_FEV_COE_REPEATS=0 would
    # survive and silently cost the run its per-depth columns.
    os.environ["TSM_FEV_COE_REPEATS"] = "1"

    args.out.mkdir(parents=True, exist_ok=True)
    want = {"model_path": str(args.model_path),
            "coe_eval_depth": args.coe_eval_depth, "eo_config": eo,
            "fev_subset": args.fev_subset,
            "task_subset_index": args.task_subset_index,
            "num_task_subsets": args.num_task_subsets}
    manifest = args.out / "manifest.json"
    if manifest.is_file():
        have = json.loads(manifest.read_text())
        diff = [k for k in want if have.get(k) != want[k]]
        if diff:
            raise SystemExit(
                f"{manifest} was written by a different run (differs in "
                f"{diff}). The alphas here would not be comparable.")
    else:
        manifest.write_text(json.dumps(want, indent=1, default=str))

    for benchmark in args.benchmarks:
        for alpha in args.alpha:
            tag = f"alpha{alpha:g}"
            dest = args.out / benchmark / tag
            csv = dest / "results.csv"
            if csv.is_file():
                logger.info("SKIP %s/%s — already scored", benchmark, tag)
                continue
            dest.mkdir(parents=True, exist_ok=True)
            logger.info("=== %s / alpha=%g", benchmark, alpha)

            forecaster = load_forecaster(SimpleNamespace(
                model_path=args.model_path, device="cuda",
                torch_dtype="float32", config_path=None,
                model_source_path=None, batch_size=args.batch_size,
                eo_max_n_variate=None, rolling_horizon=None,
                rolling_quantiles=None, coe_eval_depth=args.coe_eval_depth))
            adapter, name = make_adapter(benchmark, args)

            store = TruthStore()
            t0 = time.time()
            # NOTHING is captured at alpha = 0: `truth_feedback` installs no
            # patch there, so the store would be filled and never read — one
            # arm of the default sweep paying the whole capture cost and its
            # peak retention for a result it cannot use. Skipping it also makes
            # alpha = 0 bit-identical to a stock run down to the adapter's own
            # Dataset class, which is what the arm is for.
            capture = (contextlib.nullcontext() if alpha == 0.0
                       else capture_truth(benchmark, store))
            with capture, truth_feedback(forecaster._pipeline, alpha, store):
                df = adapter.evaluate(forecaster, output_dir=str(dest),
                                      benchmark_name=name)
            df.to_csv(csv, index=False)
            (dest / "truth.json").write_text(json.dumps(
                {"alpha": alpha, "capture_skipped": alpha == 0.0,
                 "truth_hits": store.n_hits, "truth_misses": store.n_misses,
                 "ambiguous_fingerprints": store.n_ambiguous}, indent=1))
            if alpha > 0 and store.n_hits == 0:
                raise SystemExit(
                    f"{benchmark}/{tag}: the truth never reached the model "
                    f"({store.n_misses} misses). The capture seam did not fire "
                    f"— this arm is a stock run wearing an alpha label.")
            logger.info("%s/%s: %d rows in %.1fs (truth hits=%d misses=%d "
                        "ambiguous=%d)", benchmark, tag, len(df),
                        time.time() - t0, store.n_hits, store.n_misses,
                        store.n_ambiguous)
            # Released BEFORE the next arm loads its checkpoint — `store`
            # included, or the previous arm's whole truth map is still resident
            # while the next model is read into RAM.
            forecaster = adapter = df = store = None
            gc.collect()
            if torch is not None:
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
