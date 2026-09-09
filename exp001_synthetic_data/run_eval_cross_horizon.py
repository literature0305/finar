#!/usr/bin/env python3
"""Score one EO v4 checkpoint on the equal-observed cross-horizon sweep, at
every recursion depth from 1 to --iters.

    12 tasks = 4 horizons (16/100/400/1000) x 3 dependency regimes (high/mid/low)

Every task shows the model the SAME 1040 steps of observed history, so the only
thing the horizon axis varies is the horizon. See `build_cross_horizon_trim.py`
for why the published corpus cannot do that.

DEPTH BEYOND TRAINING IS THE POINT, not an accident. `EOPipeline.report_depth`
is `max(coe_train_depth_max, production_depth(override))`, so a K=2 checkpoint
asked for 4 passes reports repeat1..repeat4. The per-iteration columns come from
`engine/evaluator.py` as `repeat{r}_MASE` / `repeat{r}_WQL`.

tsm-trainer is imported, never written to: the benchmark yaml this needs points
at the trimmed corpus, so it is generated HERE rather than added to that
checkout's configs/.

Usage
-----
    python run_eval_cross_horizon.py --model-path <eo-v4 ckpt> --out <dir> --iters 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger("finar_exp001_2")

DEFAULT_REPO = Path("/group-volume/workspace/mun-hak.lee/experiments/"
                    "tsm-trainer_001/tsm-trainer")
DEFAULT_DATA = Path("/group-volume/ts-dataset/"
                    "cross_horizon_length_trim_equal_observed")
HORIZONS = (16, 100, 400, 1000)
REGIMES = ("high", "mid", "low")


def add_repo_to_path(repo: Path) -> None:
    ev = repo / "scripts" / "forecasting" / "evaluation"
    if not ev.is_dir():
        raise SystemExit(f"no evaluation package under {repo} (looked at {ev})")
    for p in (str(ev), str(repo / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def write_benchmark_yaml(root: Path, dest: Path) -> Path:
    """The 12-task yaml, generated from what is on disk.

    Derived rather than vendored: a copy of the upstream config edited to point
    somewhere else is the thing that goes stale when a subset is added or a
    horizon changes. `offset = -H` and `prediction_length = H` are the same
    convention `configs/cross-horizon-length.yaml` uses.
    """
    import yaml

    datasets = []
    for h in HORIZONS:
        for regime in REGIMES:
            d = root / f"h{h}" / regime
            if not d.is_dir():
                raise SystemExit(
                    f"{d} is missing — run build_cross_horizon_trim.py first")
            datasets.append({
                "name": f"cross_horizon_len_eq/h{h}/{regime}",
                "data_dir": str(d), "offset": -h,
                "prediction_length": h, "num_rolls": 1,
            })
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.dump({"datasets": datasets}, sort_keys=False))
    return dest


def observed_report(repo: Path, context_length: int, patch: int) -> list[dict]:
    """Horizon vs the history the model actually gets, for both corpora.

    The point of the table: in the published corpus the stored context is 2048
    everywhere but the EFFECTIVE context is not, because
    `append_forecast_region` takes the forecast region out of the same window.
    """
    rows = []
    for h in HORIZONS:
        forecast_len = -(-h // patch) * patch
        rows.append({
            "horizon": h,
            "patch_forecast_len": forecast_len,
            "orig_stored_observed": 2048,
            "orig_effective_observed": context_length - forecast_len,
            "trim_stored_observed": 1040,
            "trim_effective_observed": min(1040, context_length - forecast_len),
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="EO v4 checkpoint dir")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--iters", type=int, default=4,
                   help="recursion depths to score, 1..N. May exceed the "
                        "checkpoint's trained depth; that is the experiment.")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.iters < 1:
        raise SystemExit("--iters must be >= 1")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    add_repo_to_path(args.repo)

    from benchmarks.chronos_bench import ChronosLiteBenchmarkAdapter  # noqa
    from run_benchmark import load_forecaster                          # noqa

    args.out.mkdir(parents=True, exist_ok=True)
    yaml_path = write_benchmark_yaml(args.data, args.out / "tasks.yaml")
    logger.info("12-task yaml -> %s", yaml_path)

    forecaster = load_forecaster(SimpleNamespace(
        model_path=args.model_path, device="cuda", torch_dtype="float32",
        config_path=None, model_source_path=None, batch_size=args.batch_size,
        eo_max_n_variate=None, rolling_horizon=None, rolling_quantiles=None,
        coe_eval_depth=args.iters))
    eo = getattr(getattr(forecaster, "_pipeline", None), "model", None)
    cfg = getattr(eo, "eo_config", None)
    if cfg is None:
        raise SystemExit(f"{args.model_path} is not an EO model")
    trained = int(getattr(cfg, "coe_train_depth_max", 1))
    ctx_len = int(getattr(cfg, "context_length", 0))
    patch = int(getattr(cfg, "patch_size", 0) or 0)
    logger.info("checkpoint: trained_depth=%d context_length=%d patch=%d; "
                "scoring depths 1..%d", trained, ctx_len, patch, args.iters)
    if args.iters > trained:
        logger.info("depths %d..%d are DEEPER than this checkpoint was trained "
                    "for — recorded, and marked in the table",
                    trained + 1, args.iters)

    # The confound this corpus removes only stays removed if the model's window
    # is the one it was cut for. Reported, not enforced.
    for row in observed_report(args.repo, ctx_len, patch):
        if row["trim_effective_observed"] != row["trim_stored_observed"]:
            logger.warning(
                "H=%d: this model leaves %d steps of context, below the %d "
                "stored, so this subset alone is truncated further",
                row["horizon"], row["trim_effective_observed"],
                row["trim_stored_observed"])

    adapter = ChronosLiteBenchmarkAdapter(
        config_path=yaml_path, batch_size=args.batch_size)
    df = adapter.evaluate(forecaster)
    dest = args.out / "results.csv"
    df.to_csv(dest, index=False)
    (args.out / "manifest.json").write_text(json.dumps({
        "model_path": str(args.model_path), "iters": args.iters,
        "trained_depth": trained, "context_length": ctx_len,
        "patch_size": patch, "data": str(args.data),
        "observed_report": observed_report(args.repo, ctx_len, patch),
    }, indent=1))
    logger.info("wrote %s (%d rows)", dest, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
