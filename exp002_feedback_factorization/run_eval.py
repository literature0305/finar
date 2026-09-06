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


def require_coe(model_path: str, depth: int) -> dict:
    """Refuse a checkpoint this experiment is not defined for.

    Three separate reasons, each of which otherwise fails silently or
    misleadingly:

      * no eo_config      -> not an EO v4 model, no recursion at all
      * coe_bottleneck    -> passes exchange hidden state, so there is no
                             per-variate forecast feedback to restrict and the
                             whole experiment is undefined (work order item 15)
      * K < 2             -> iteration 2 is extrapolation past training, not
                             refinement; and _repeats_worth_asking returns False
                             before it reads TSM_FEV_COE_REPEATS, so the depth
                             columns would simply be absent with no explanation
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
    k = eo.get("coe_train_depth_max")
    if not isinstance(k, int) or k < 2:
        raise SystemExit(
            f"coe_train_depth_max={k!r} < 2: this checkpoint was not trained to "
            f"refine, so iteration 2 would be extrapolation rather than the "
            f"refinement this experiment measures.")
    if depth > k:
        logger.warning("--coe-eval-depth %d exceeds coe_train_depth_max=%d — "
                       "depths above %d are outside the trained range",
                       depth, k, k)
    logger.info("checkpoint: K=%s eval_depth=%s bottleneck=%s residual=%s "
                "feedback_drop=%s", k, eo.get("coe_eval_depth"),
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
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.coe_eval_depth < 2:
        raise SystemExit("--coe-eval-depth must be >= 2; the experiment is an "
                         "iteration-1 vs iteration-2 comparison")

    eo = require_coe(args.model_path, args.coe_eval_depth)
    add_repo_to_path(args.repo)

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
            num_workers=args.num_workers,
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
