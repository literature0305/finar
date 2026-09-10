#!/usr/bin/env python3
"""Score one EO v4 checkpoint on fev-bench's covariate subset, three ways.

Runs the SAME checkpoint under three feedback regimes — full / self_only /
cov_only (see feedback_patch.py) — at every recursion depth, so iteration 1 and
iteration 2 can be compared per regime and per covariate subset.

Iteration 1 is not a fourth run. It is identical in all three regimes by
construction: the regimes only change what the write-back hands to pass 2, and
pass 1 starts from zero everywhere regardless. `build_table.py` therefore reads
the no-feedback baseline off `repeat1_*` of the `full` run, and asserts the
other two agree with it — a disagreement means the patch leaked into pass 1.

tsm-trainer is imported, never written to. `run_benchmark.load_forecaster` and
`FevBenchAdapter` are used as-is; the only thing this adds is the patch, which
is installed around the adapter call and removed after.

Usage
-----
    python run_eval.py --model-path <ckpt> --repo <tsm-trainer> --out <dir>
    python run_eval.py ... --scenarios full self_only     # a subset
    python run_eval.py ... --coe-eval-depth 2             # default 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# CPU CAP — AT MODULE SCOPE, ABOVE numpy/torch. OpenMP and BLAS size their pools
# when the library first loads, so a cap applied after those imports is ignored
# for them. `main()` calls limit_cpu again once --num-workers is parsed; that
# second call is runtime-only work (torch, pyarrow, fev) and is order-free.
sys.path.append(str(next(d for d in Path(__file__).resolve().parents
                         if (d / "finar_cpu.py").is_file())))
from finar_cpu import argv_workers, limit_cpu  # noqa: E402

limit_cpu(argv_workers(), quiet=True)

logger = logging.getLogger("finar_exp002")

DEFAULT_REPO = Path("/group-volume/workspace/mun-hak.lee/experiments/"
                    "tsm-trainer_001/tsm-trainer")


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
    `init_warmup_depth_max`. tsm-trainer's own config documents the
    interaction, and the default is the trap —

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


def trained_depth(eo: dict) -> tuple[int, int, int]:
    """``(passes the model was trained to produce, K, warm-up depth)``.

    The scored chain runs `K` passes on top of a warm-up of `t` grad-free
    passes, so a checkpoint at `K = 1, t = 1` has been trained on an input that
    one pass already refined — which is what makes evaluating at depth 2 a
    depth it has seen, and is exactly the case a bare `coe_train_depth_max < 2`
    test rejects by mistake.
    """
    k = eo.get("coe_train_depth_max")
    k = k if isinstance(k, int) and k >= 1 else 1
    t = warmup_depth(eo)
    return k + t, k, t


def require_coe(model_path: str, depth: int) -> dict:
    """Refuse a checkpoint this experiment is not defined for.

    Three separate reasons, each of which otherwise fails silently or
    misleadingly:

      * no eo_config      -> not an EO v4 model, no recursion at all
      * coe_bottleneck    -> passes exchange hidden state, so there is no
                             per-variate forecast feedback to restrict and the
                             whole experiment is undefined (work order item 15)
      * trained depth < 2 -> iteration 2 is extrapolation past training, not
                             refinement; and _repeats_worth_asking returns False
                             before it reads TSM_FEV_COE_REPEATS, so the depth
                             columns would simply be absent with no explanation

    The trained depth is `coe_train_depth_max` PLUS the grad-free warm-up (see
    `trained_depth`), not `coe_train_depth_max` alone: a checkpoint trained at
    K=1 with a warm-up of 1 has been trained on refined inputs and is a
    legitimate subject here, and testing K alone would refuse it.
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
            f"the bottleneck the passes chain in hidden space and never write a "
            f"forecast back, so 'feed back only the targets' has no meaning. "
            f"This experiment is undefined for this checkpoint.")
    trained, k, warm = trained_depth(eo)
    if trained < 2:
        raise SystemExit(
            f"trained depth {trained} < 2 (coe_train_depth_max={k!r}, grad-free "
            f"warm-up {warm}): this checkpoint was not trained to refine, so "
            f"iteration 2 would be extrapolation rather than the refinement "
            f"this experiment measures.\n"
            f"  init_from_repeat_prediction="
            f"{eo.get('init_from_repeat_prediction')!r} "
            f"init_warmup_depth_max={eo.get('init_warmup_depth_max')!r}\n"
            f"  Note that the switch alone is not enough: with "
            f"init_warmup_depth_max null the warm-up follows "
            f"coe_train_depth_max and is INERT at 1, so it reports true while "
            f"doing nothing. Set init_warmup_depth_max >= 1 to train it.")
    if warm and k < 2:
        # Trained, but not identically to a K>=2 checkpoint. Said once, here,
        # rather than left for whoever reads the table to wonder about.
        logger.warning(
            "trained depth %d comes from a grad-free WARM-UP (K=%d + warm-up "
            "%d), not from a scored chain of %d passes. Three differences "
            "survive: the warm-up carries no gradient; the scored chain "
            "restarts its coe_repeat_encoding labels at 0, so evaluating pass "
            "2 labels it 1 where training labelled it 0; and with "
            "init_warmup_stochastic=%r the depth is drawn per step, so some "
            "steps trained no warm-up at all.",
            trained, k, warm, trained, eo.get("init_warmup_stochastic"))
    if depth > trained:
        logger.warning("--coe-eval-depth %d exceeds the trained depth %d — "
                       "depths above %d are outside the trained range",
                       depth, trained, trained)
    logger.info("checkpoint: trained_depth=%s (K=%s + warm-up %s) eval_depth=%s "
                "bottleneck=%s residual=%s feedback_drop=%s",
                trained, k, warm, eo.get("coe_eval_depth"),
                eo.get("coe_bottleneck"), eo.get("coe_residual"),
                eo.get("feedback_variable_drop_max_ratio"))
    return eo


#: Where a fev parquet cache conventionally lives on these hosts.
_LOCAL_FEV = Path("/group-volume/ts-dataset/benchmarks/fev_bench")


def resolve_fev_data(explicit):
    """The fev data root: explicit, else the local cache, else None.

    None makes the adapter take fev's Hub code path, which needs network for
    EVERY task — measured here as the run's bottleneck, and a hard failure on a
    host without egress. The local cache is preferred whenever it is present,
    and the choice is logged so a slow run is never a mystery.
    """
    if explicit:
        return explicit
    if _LOCAL_FEV.is_dir():
        logger.info("fev data: %s (local parquet, offline)", _LOCAL_FEV)
        return str(_LOCAL_FEV)
    logger.warning("fev data: no local cache at %s — falling back to the HF "
                   "Hub, which needs network for every task", _LOCAL_FEV)
    return None


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO,
                   help="tsm-trainer checkout to IMPORT from (read-only)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fev-data", default=None,
                   help="fev parquet root. Default: the conventional local "
                        "cache if it exists, else None (HF Hub over the "
                        "network). See resolve_fev_data.")
    p.add_argument("--scenarios", nargs="+", default=None,
                   help="subset of full/self_only/cov_only")
    p.add_argument("--coe-eval-depth", type=int, default=2,
                   help="recursion depth to score at; every depth 1..N is "
                        "reported. Default 2 — the comparison is iter1 vs iter2")
    p.add_argument("--batch-size", type=int, default=32)
    # ALSO the CPU cap for this process, not only the adapter's worker count —
    # see finar_cpu. One number, so the two cannot disagree.
    p.add_argument("--num-workers", type=int, default=None,
                   help="CPU threads this run may use AND the adapter's worker "
                        "count (default: $OMP_NUM_THREADS, else 8)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.coe_eval_depth < 2:
        raise SystemExit("--coe-eval-depth must be >= 2; the experiment is an "
                         "iteration-1 vs iteration-2 comparison")

    eo = require_coe(args.model_path, args.coe_eval_depth)
    add_repo_to_path(args.repo)
    cap = limit_cpu(args.num_workers)
    # NOT the cap. `num_workers` sizes fev's metric thread pool, and each of
    # those threads is itself entitled to `cap` BLAS threads — passing the cap
    # here would make --num-workers 32 mean 32x32, multiplying the concurrency
    # the flag exists to bound.
    adapter_workers = max(1, min(4, cap))

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from feedback_patch import SCENARIOS, feedback_scenario  # noqa: E402

    scenarios = args.scenarios or list(SCENARIOS)
    bad = [s for s in scenarios if s not in SCENARIOS]
    if bad:
        raise SystemExit(f"unknown scenario(s) {bad}; known: {list(SCENARIOS)}")

    # Assigned, NOT setdefault: an inherited TSM_FEV_COE_REPEATS=0 in the
    # calling shell would survive a setdefault and silently cost the whole run
    # its depth columns, which this script would only notice afterwards.
    os.environ["TSM_FEV_COE_REPEATS"] = "1"

    from types import SimpleNamespace  # noqa: E402

    from benchmarks.fev_bench import FevBenchAdapter  # noqa: E402
    from run_benchmark import load_forecaster  # noqa: E402

    fev_data = resolve_fev_data(args.fev_data)

    args.out.mkdir(parents=True, exist_ok=True)
    # A MANIFEST, checked before the per-scenario resume below. The three
    # scenarios are only comparable if they came from the same checkpoint at
    # the same depth; without this, re-running a finished directory with a
    # different --model-path would SKIP the scenario already on disk and score
    # the rest with the new model, and the table would read a checkpoint
    # difference as a feedback-regime effect.
    manifest = {"model_path": str(Path(args.model_path).resolve()),
                "coe_eval_depth": args.coe_eval_depth, "eo_config": eo}
    mpath = args.out / "manifest.json"
    if mpath.is_file():
        prev = json.loads(mpath.read_text())
        diff = [k for k in ("model_path", "coe_eval_depth", "eo_config")
                if prev.get(k) != manifest[k]]
        if diff:
            raise SystemExit(
                f"{mpath} was written by a different run (differs in {diff}). "
                f"The scenarios in this directory would not be comparable. Use "
                f"a fresh --out, or delete it to rescore from scratch.")
    mpath.write_text(json.dumps(manifest, indent=1))

    for scenario in scenarios:
        dest = args.out / scenario
        if (dest / "fev_bench.csv").is_file():
            logger.info("SKIP %s — already scored", scenario)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("=== scenario %s", scenario)

        # A FRESH forecaster per scenario. The patch is removed on exit, but a
        # model carries pipeline state (stashes, caches) and reusing one across
        # regimes would make an ordering effect indistinguishable from a
        # scenario effect — the whole result is a comparison between them.
        forecaster = load_forecaster(SimpleNamespace(
            model_path=args.model_path, device="cuda", torch_dtype="float32",
            config_path=None, model_source_path=None,
            batch_size=args.batch_size, eo_max_n_variate=None,
            rolling_horizon=None, rolling_quantiles=None,
            coe_eval_depth=args.coe_eval_depth,
        ))
        adapter = FevBenchAdapter(
            data_dir=fev_data, batch_size=args.batch_size,
            num_workers=adapter_workers,
            # The covariate subset IS the experiment: self_only and cov_only
            # are only defined where there are covariates to withhold.
            subset="covariate",
        )
        t0 = time.time()
        with feedback_scenario(forecaster._pipeline, scenario):
            df = adapter.evaluate(forecaster, output_dir=str(dest),
                                  benchmark_name="fev_bench")
        df.to_csv(dest / "fev_bench.csv", index=False)
        depths = sorted({c for c in df.columns if c.startswith("repeat")})
        logger.info("  %d rows in %.1fs; depth columns: %s",
                    len(df), time.time() - t0, depths or "NONE")
        if not depths:
            logger.warning(
                "  no repeat<d> columns — the run reported a single depth, so "
                "there is no iteration axis. Check coe_eval_depth.")
        del forecaster
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
