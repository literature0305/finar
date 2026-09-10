#!/usr/bin/env python3
"""Score one EO v4 checkpoint on the MULTIVARIATE benchmarks, three ways.

Runs the SAME checkpoint under three feedback regimes — full / self_only /
cov_only (see feedback_patch.py) — at every recursion depth, on two benchmarks
that carry multivariate targets and no covariates:

    fev_mul    fev-bench's `multivariate` subset (>1 target, no dynamic covs)
    gift_mul   GIFT-Eval HF, multivariate tasks fed whole (group attention on)

Neither names a target, so `feedback_patch` DESIGNATES the first variate of
each group as the target and treats the rest as pseudo-covariates. "Self" is
then exactly one variate's own trajectory and "cross" is exactly the others —
which is the split the covariate version of this experiment cannot make on a
multi-target task, because it feeds back every target under `self_only`.

Iteration 1 is not a fourth run. It is identical in all three regimes by
construction: the regimes only change what the write-back hands to pass 2, and
pass 1 starts from zero everywhere regardless. `build_table.py` therefore reads
the no-feedback baseline off `repeat1_*` of the `full` run and asserts the
other two agree with it — a disagreement means the patch leaked into pass 1.

THE DESIGNATION IS AUDITED, NOT ASSUMED. Each run records, per task actually
fed to the model, `(n_variates, target_variate, n_past_covariates)`. The three
scenarios must produce identical records; this driver compares them and refuses
to leave a directory whose arms disagree, because a target that moved between
arms would make the comparison meaningless without making it look wrong.

tsm-trainer is imported, never written to. `run_benchmark.load_forecaster`,
`FevBenchAdapter` and `GiftEvalHFMulAdapter` are used as-is; the only thing
this adds is the patch, installed around the adapter call and removed after.

Usage
-----
    python run_eval.py --model-path <ckpt> --repo <tsm-trainer> --out <dir>
    python run_eval.py ... --benchmarks fev_mul          # one benchmark
    python run_eval.py ... --scenarios full self_only    # a subset
    python run_eval.py ... --coe-eval-depth 2            # default 2
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

try:
    import torch
except ImportError:          # only needed to release VRAM between scenarios
    torch = None

logger = logging.getLogger("finar_exp002_pseudo")

DEFAULT_REPO = Path("/group-volume/workspace/mun-hak.lee/experiments/"
                    "tsm-trainer_001/tsm-trainer")

#: Where the two benchmarks' data conventionally lives on these hosts.
_LOCAL_FEV = Path("/group-volume/ts-dataset/benchmarks/fev_bench")
_LOCAL_GIFT = Path("/group-volume/ts-dataset/benchmarks/gift_eval")

BENCHMARKS = ("fev_mul", "gift_mul")


def add_repo_to_path(repo: Path) -> None:
    ev = repo / "scripts" / "forecasting" / "evaluation"
    if not ev.is_dir():
        raise SystemExit(f"no evaluation package under {repo} (looked at {ev})")
    for p in (str(ev), str(repo / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def warmup_depth(eo: dict) -> int:
    """How deep the grad-free warm-up runs before the scored chain.

    `init_from_repeat_prediction` ALONE does not mean the model was trained to
    refine: it is only the switch, and the depth lives in
    `init_warmup_depth_max`, whose default is the trap —

        null (default)  follow coe_train_depth_max, and stay INERT when that
                        is 1
        0               no warm-up, without touching the switch
        n >= 1          draw from [0, n] whatever coe_train_depth_max is

    — so `coe_train_depth_max: 1` with `init_warmup_depth_max: null` leaves the
    switch reading `true` while the warm-up does nothing at all.
    """
    # `coe_bottleneck` is part of the gate, not a separate concern: without the
    # bottleneck the passes chain in hidden space and the warm-up never runs at
    # all. tsm-trainer folds all three conditions into
    # `AEDForecastingConfig.warmup_active`, whose docstring records that the
    # model and the config had already drifted apart once by writing the switch
    # and the bottleneck out at two sites — and that a `coe_bottleneck: false`
    # run was consequently told its eval depth was covered by a warm-up that
    # never happened. Omitting it here would admit exactly that checkpoint.
    if not (eo.get("init_from_repeat_prediction") and eo.get("coe_bottleneck")):
        return 0
    d = eo.get("init_warmup_depth_max")
    if d is None:
        k = eo.get("coe_train_depth_max")
        k = k if isinstance(k, int) else 1
        return 0 if k <= 1 else k          # inert at K=1, by tsm-trainer's rule
    return int(d) if isinstance(d, int) and d > 0 else 0


def require_coe(model_path: str, depth: int) -> dict:
    """Refuse a checkpoint this experiment is not defined for.

    Three separate reasons, each of which otherwise fails silently or
    misleadingly:

      * no eo_config      -> not an EO v4 model, no recursion at all
      * coe_bottleneck    -> passes exchange hidden state, so there is no
                             per-variate forecast feedback to restrict and the
                             whole experiment is undefined
      * trained depth < 2 -> iteration 2 is extrapolation past training, not
                             refinement; and `_repeats_worth_asking` returns
                             False before it reads TSM_FEV_COE_REPEATS, so the
                             depth columns would simply be absent, unexplained

    The trained depth is `coe_train_depth_max` PLUS the grad-free warm-up, not
    `coe_train_depth_max` alone: a checkpoint trained at K=1 with a warm-up of 1
    has been trained on refined inputs and is a legitimate subject here.
    """
    cfg = Path(model_path) / "config.json"
    if not cfg.is_file():
        raise SystemExit(f"{cfg} not found — this needs a local EO v4 checkpoint")
    eo = (json.loads(cfg.read_text()).get("eo_config") or {})
    if not eo:
        raise SystemExit(f"{model_path} has no eo_config — not an EO v4 model")
    if eo.get("coe_bottleneck") is not True:
        raise SystemExit(
            f"coe_bottleneck={eo.get('coe_bottleneck')!r}, not True. Without "
            f"the bottleneck the passes chain in hidden space and never write "
            f"a forecast back, so 'feed back only the target' has no meaning.")
    k = eo.get("coe_train_depth_max")
    k = k if isinstance(k, int) and k >= 1 else 1
    warm = warmup_depth(eo)
    trained = k + warm
    if trained < 2:
        raise SystemExit(
            f"trained depth {trained} < 2 (coe_train_depth_max={k}, grad-free "
            f"warm-up {warm}): this checkpoint was not trained to refine, so "
            f"iteration 2 would be extrapolation rather than the refinement "
            f"this experiment measures.\n"
            f"  init_from_repeat_prediction="
            f"{eo.get('init_from_repeat_prediction')!r} "
            f"init_warmup_depth_max={eo.get('init_warmup_depth_max')!r}\n"
            f"  The switch alone is not enough: with init_warmup_depth_max "
            f"null the warm-up follows coe_train_depth_max and is INERT at 1.")
    if warm and k < 2:
        logger.warning(
            "trained depth %d comes from a grad-free WARM-UP (K=%d + warm-up "
            "%d). The warm-up carries no gradient, the scored chain restarts "
            "its coe_repeat_encoding labels at 0, and with "
            "init_warmup_stochastic=%r the depth is drawn per step — so some "
            "steps trained no warm-up at all.",
            trained, k, warm, eo.get("init_warmup_stochastic"))
    if depth > trained:
        logger.warning("--coe-eval-depth %d exceeds the trained depth %d",
                       depth, trained)
    logger.info("checkpoint: trained_depth=%s (K=%s + warm-up %s) "
                "eval_depth=%s bottleneck=%s residual=%s",
                trained, k, warm, eo.get("coe_eval_depth"),
                eo.get("coe_bottleneck"), eo.get("coe_residual"))
    return eo


def resolve_data(explicit, default: Path, what: str):
    """The data root: explicit, else the local cache, else None (Hub path)."""
    if explicit:
        return str(explicit)
    if default.is_dir():
        logger.info("%s data: %s (local, offline)", what, default)
        return str(default)
    logger.warning("%s data: no local cache at %s — falling back to the Hub, "
                   "which needs network", what, default)
    return None


def gift_multivariate_datasets() -> list[str]:
    """The GIFT-Eval datasets whose targets are multivariate.

    THE ADAPTER DOES NOT DO THIS FOR US, and its name says otherwise. The
    `_mul` in `GiftEvalHFMulAdapter` is a HANDLING mode — "feed a multivariate
    dataset whole through group attention instead of the official
    `to_univariate=True` flattening" — not a subset. Constructed without
    `datasets=`, its `load_tasks()` returns `ALL_DATASETS x terms`: all 97
    tasks, 54 of them univariate.

    Those 54 are not merely uninformative here, they are DEGENERATE BY
    CONSTRUCTION: with one variate there is no "other", so `self_only` feeds
    back everything (identical to `full`) and `cov_only` feeds back nothing
    (identical to iteration 1). They also cost most of the run — 82% of the
    benchmark's total `items x horizon` sits in univariate tasks, and the three
    largest (`electricity/15T/{short,medium,long}`, 7,400 items each) are all
    univariate.

    Read from tsm-trainer's committed `baselines/gift_eval_hf_variate_types.csv`
    rather than by opening each corpus. `get_gift_eval_hf_variate_types()` is
    NOT used: it WRITES the csv back when it meets an unknown name, and that
    checkout is read-only here. A name the csv does not cover is therefore a
    hard failure — dropping it silently would shrink the suite invisibly, which
    is the same class of mistake this whole experiment exists to avoid.
    """
    import pandas as pd
    from benchmarks import variate_types
    from benchmarks.gift_eval_hf import ALL_DATASETS

    df = pd.read_csv(variate_types._GIFT_EVAL_HF_CSV)
    known = set(df["dataset"])
    unknown = [d for d in ALL_DATASETS if d not in known]
    if unknown:
        raise SystemExit(
            f"{variate_types._GIFT_EVAL_HF_CSV} does not cover {unknown}, so "
            f"their variate count is unknown and they would be dropped "
            f"silently. Populate the csv in the tsm-trainer checkout first.")
    keep = [d for d in ALL_DATASETS
            if d in set(df.loc[df["is_multivariate"].astype(bool), "dataset"])]
    if not keep:
        raise SystemExit("no multivariate GIFT-Eval datasets found")
    logger.info("gift_mul: %d of %d datasets are multivariate", len(keep),
                len(ALL_DATASETS))
    return keep


def make_adapter(benchmark: str, args):
    """The stock adapter for one benchmark, restricted to MULTIVARIATE tasks.

    `ignore_group_id=False` in both: it is what feeds a task's variates as one
    group instead of flattening them into independent univariate series. With
    it True there would be no cross-variate feedback to restrict and all three
    scenarios would return the same numbers.

    Both arms are filtered, and they are filtered DIFFERENTLY because the two
    adapters expose the subset differently: fev-bench names it (`subset=`),
    GIFT-Eval takes an explicit dataset list (`datasets=`). Leaving the second
    off is not a smaller filter, it is no filter at all.
    """
    slice_kw = {"task_subset_index": args.task_subset_index,
                "num_task_subsets": args.num_task_subsets}
    if benchmark == "fev_mul":
        from benchmarks.fev_bench import FevBenchAdapter
        return FevBenchAdapter(
            data_dir=resolve_data(args.fev_data, _LOCAL_FEV, "fev"),
            subset="multivariate", batch_size=args.batch_size,
            ignore_group_id=False, **slice_kw), "fev_bench"
    if benchmark == "gift_mul":
        from benchmarks.gift_eval_hf_mul import GiftEvalHFMulAdapter
        return GiftEvalHFMulAdapter(
            data_dir=resolve_data(args.gift_data, _LOCAL_GIFT, "gift"),
            datasets=gift_multivariate_datasets(),
            batch_size=args.batch_size, ignore_group_id=False,
            name="gift_eval_hf_mul", **slice_kw), "gift_eval_hf_mul"
    raise SystemExit(f"unknown benchmark {benchmark!r}; known: {BENCHMARKS}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="EO v4 checkpoint dir")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--coe-eval-depth", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=None,
                   help="CPU threads this run may use (default: "
                        "$OMP_NUM_THREADS, else 8). Caps torch, "
                        "pyarrow and fev; the launcher exports the "
                        "OpenMP/BLAS vars, which must be set before "
                        "python starts to take effect.")
    p.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS),
                   help=f"subset of {BENCHMARKS}")
    p.add_argument("--scenarios", nargs="+", default=None,
                   help="subset of full/self_only/cov_only")
    p.add_argument("--fev-data", default=None)
    p.add_argument("--gift-data", default=None)
    # The adapters' own task slicing, exposed because GIFT-Eval does not fit in
    # 23 GB in one pass on these hosts. Slicing changes WHICH tasks a run
    # scores, so the slice is part of what makes two runs comparable — it goes
    # in the manifest with the checkpoint and the depth.
    p.add_argument("--task-subset-index", type=int, default=0)
    p.add_argument("--num-task-subsets", type=int, default=1)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    if args.coe_eval_depth < 2:
        raise SystemExit(
            f"--coe-eval-depth {args.coe_eval_depth} reports a single depth, "
            f"and this experiment is an iteration-1 vs iteration-2 comparison.")
    bad = [b for b in args.benchmarks if b not in BENCHMARKS]
    if bad:
        raise SystemExit(f"unknown benchmark(s) {bad}; known: {list(BENCHMARKS)}")

    # The script's OWN directory first: `add_repo_to_path` puts tsm-trainer's
    # evaluation/ and src/ on sys.path, and `feedback_patch` is a sibling of
    # this file. Relying on the interpreter having put the script directory
    # there is relying on how the script was invoked.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    add_repo_to_path(args.repo)
    # BEFORE any adapter or model is built. Every finar experiment
    # drives the adapters itself, so run_benchmark.main()'s
    # torch/pyarrow/fev caps never run for it — only its module-level
    # env-var pass does. See finar_cpu.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from finar_cpu import limit_cpu  # noqa: E402
    limit_cpu(args.num_workers)
    from feedback_patch import (  # noqa: E402
        SCENARIOS, Designation, feedback_scenario)

    scenarios = args.scenarios or list(SCENARIOS)
    bad = [s for s in scenarios if s not in SCENARIOS]
    if bad:
        raise SystemExit(f"unknown scenario(s) {bad}; known: {list(SCENARIOS)}")

    eo = require_coe(args.model_path, args.coe_eval_depth)

    from types import SimpleNamespace  # noqa: E402

    from run_benchmark import load_forecaster  # noqa: E402

    # ASSIGNED, not setdefault: an inherited TSM_FEV_COE_REPEATS=0 in the
    # calling shell would survive a setdefault and silently cost the whole run
    # its depth columns, which this script would only notice afterwards. The
    # fev adapter gates its depth sweep on this; harmless elsewhere.
    os.environ["TSM_FEV_COE_REPEATS"] = "1"

    args.out.mkdir(parents=True, exist_ok=True)
    # A MANIFEST, checked before the per-scenario resume below. The arms are
    # only comparable if they came from the same checkpoint at the same depth,
    # and without this a re-run with a different --model-path would SKIP the
    # scenario already on disk and score the rest with the new one.
    manifest = args.out / "manifest.json"
    want = {"model_path": str(args.model_path),
            "coe_eval_depth": args.coe_eval_depth, "eo_config": eo,
            "task_subset_index": args.task_subset_index,
            "num_task_subsets": args.num_task_subsets}
    if manifest.is_file():
        have = json.loads(manifest.read_text())
        # Over `want`'s own keys, not a second hand-written list: a field
        # added to the manifest but forgotten here would silently stop being
        # compared, which is exactly what this guard exists to prevent.
        diff = [k for k in want if have.get(k) != want[k]]
        if diff:
            raise SystemExit(
                f"{manifest} was written by a different run (differs in "
                f"{diff}). The arms in this directory would not be comparable. "
                f"Use a fresh --out, or delete it.")
    else:
        manifest.write_text(json.dumps(want, indent=1, default=str))

    audits: dict[tuple[str, str], list] = {}
    for benchmark in args.benchmarks:
        for scenario in scenarios:
            dest = args.out / benchmark / scenario
            csv = dest / "results.csv"
            if csv.is_file():
                logger.info("SKIP %s/%s — already scored", benchmark, scenario)
                continue
            dest.mkdir(parents=True, exist_ok=True)
            logger.info("=== %s / %s", benchmark, scenario)

            # A FRESH forecaster per scenario. The patch is removed on exit,
            # but a model that has already run under one regime carries its
            # caches; reusing it would confound the scenario effect, which is
            # the whole result.
            forecaster = load_forecaster(SimpleNamespace(
                model_path=args.model_path, device="cuda",
                torch_dtype="float32", config_path=None,
                model_source_path=None, batch_size=args.batch_size,
                eo_max_n_variate=None, rolling_horizon=None,
                rolling_quantiles=None,
                coe_eval_depth=args.coe_eval_depth))
            adapter, name = make_adapter(benchmark, args)

            audit = Designation()
            t0 = time.time()
            with feedback_scenario(forecaster._pipeline, scenario, audit=audit):
                df = adapter.evaluate(forecaster, output_dir=str(dest),
                                      benchmark_name=name)
            df.to_csv(csv, index=False)
            audits[(benchmark, scenario)] = audit
            logger.info("%s/%s: %d rows in %.1fs -> %s",
                        benchmark, scenario, len(df), time.time() - t0, csv)
            # RELEASED before the next scenario loads its own. Two complete
            # models were otherwise resident at every transition — ~0.5 GB of
            # VRAM and as much host RAM — on a host that has already been
            # OOM-killed by this sweep. `gc.collect` is needed rather than
            # optional: the write-back patch makes a model -> bound method ->
            # closure -> model cycle during the run.
            forecaster = adapter = df = None
            gc.collect()
            if torch is not None:
                torch.cuda.empty_cache()

    # THE DESIGNATION CHECK. Compared, not asserted: a target that moved
    # between arms would leave three plausible tables that are not measuring
    # the same thing, and nothing else in the pipeline would notice.
    for benchmark in args.benchmarks:
        seen = {sc: audits[(benchmark, sc)] for sc in scenarios
                if (benchmark, sc) in audits}
        if len(seen) < 2:
            continue
        ref_sc, ref = next(iter(seen.items()))
        for sc, d in seen.items():
            if d.signature() != ref.signature():
                raise SystemExit(
                    f"{benchmark}: the target designation differs between "
                    f"{ref_sc!r} ({ref.signature()}) and {sc!r} "
                    f"({d.signature()}). The arms are not measuring the same "
                    f"quantity.")
        # ONE file per benchmark, after the check. The per-scenario copies it
        # replaced were written three times precisely because they are
        # identical — which is the thing the check has just established.
        summary = ref.to_dict()
        summary["arms_agreeing"] = sorted(seen)
        (args.out / benchmark / "designation.json").write_text(
            json.dumps(summary, indent=1))
        logger.info("%s: designation identical across %d arm(s), %d tasks, "
                    "group widths %d..%d (%d univariate — degenerate, both "
                    "regimes collapse there)", benchmark, len(seen),
                    summary["n_tasks_seen"], summary["width_min"],
                    summary["width_max"], summary["n_univariate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
