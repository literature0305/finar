#!/usr/bin/env python3
"""Score one EO v4 checkpoint on the equal-observed cross-horizon sweep, at
every recursion depth from 1 to --coe-eval-depth.

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
    python run_eval_cross_horizon.py --model-path <eo-v4 ckpt> --out <dir> --coe-eval-depth 4
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger("finar_exp001_2")

DEFAULT_REPO = Path("/group-volume/workspace/mun-hak.lee/experiments/"
                    "tsm-trainer_001/tsm-trainer")
DEFAULT_DATA = Path("/group-volume/ts-dataset/"
                    "cross_horizon_length_trim_equal_observed")
# From the sibling that BUILT the corpus and the sibling that already knows how
# to read a checkpoint. Imported, not restated: a `--observed` or a fifth
# horizon that moved only one of two copies would leave this file describing a
# corpus that is not on disk, with no error.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cross_horizon_trim import (HORIZONS, REGIMES,  # noqa: E402
                                      SOURCE_OBSERVED, effective_observed)
from run_eval import add_repo_to_path, checkpoint_depths  # noqa: E402


def write_benchmark_yaml(root: Path, dest: Path) -> Path:
    """The 12-task yaml, generated from what is on disk.

    ENUMERATED FROM DISK, so a subset added to the corpus is picked up rather
    than silently ignored — which is what iterating a hardcoded list did, while
    the docstring claimed otherwise. `offset = -H` and `prediction_length = H`
    are the same convention `configs/cross-horizon-length.yaml` uses.

    Per-entry `data_dir` rather than a shared `datasets_root` (which
    `run_eval.py::write_config` uses) so the yaml is self-contained: the
    results directory can be read on a host where the corpus sits elsewhere.
    """
    import yaml

    datasets = []
    for d in sorted(root.glob("h*/*")):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"h(\d+)", d.parent.name)
        if not m or d.name not in REGIMES:
            continue
        h = int(m.group(1))
        datasets.append({
            "name": f"cross_horizon_len_eq/h{h}/{d.name}",
            "data_dir": str(d), "offset": -h,
            "prediction_length": h, "num_rolls": 1,
        })
    if not datasets:
        raise SystemExit(
            f"no h<N>/<regime> subsets under {root} — run "
            f"build_cross_horizon_trim.py first")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.dump({"datasets": datasets}, sort_keys=False))
    return dest


def corpus_meta(data: Path) -> dict:
    """What the corpus says it is, from the file the build writes for this.

    Read rather than assumed. `build_cross_horizon_trim.py` records
    `observed_length` and the window it was cut for, and calls that file "the
    only on-disk statement of what was cut" — but every consumer used to
    hardcode 1040, so a corpus built with `--observed 900` produced a manifest,
    a csv and a figure that all still claimed 1040.
    """
    m = data / "metadata.json"
    if not m.is_file():
        raise SystemExit(
            f"{m} is missing, so the observed length this corpus was cut to is "
            f"unknown — run build_cross_horizon_trim.py, or point --data-root "
            f"at a corpus it built")
    d = json.loads(m.read_text())
    if "observed_length" not in d:
        raise SystemExit(f"{m} does not record observed_length")
    return d


def observed_report(stored: int, context_length: int, patch: int) -> list[dict]:
    """Horizon vs the history the model actually gets, for both corpora.

    The point of the table: in the published corpus the stored context is 2048
    everywhere but the EFFECTIVE context is not, because
    `append_forecast_region` takes the forecast region out of the same window.
    """
    return [{
        "horizon": h,
        "patch_forecast_len": -(-h // patch) * patch,
        "orig_stored_observed": SOURCE_OBSERVED,
        "orig_effective_observed": effective_observed(h, context_length, patch),
        "trim_stored_observed": stored,
        "trim_effective_observed": min(
            stored, effective_observed(h, context_length, patch)),
    } for h in HORIZONS]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="EO v4 checkpoint dir")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    # `--coe-eval-depth`, spelled the way run_benchmark.py and every other
    # experiment's run_eval.py spell it, so one name means one thing across the
    # repo. The shell wrapper's `--depth` maps onto it, as it does in exp001..5.
    p.add_argument("--coe-eval-depth", type=int, default=4, dest="depth",
                   help="recursion depths to score, 1..N. May exceed the "
                        "checkpoint's trained depth; that is the experiment.")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.depth < 1:
        raise SystemExit("--coe-eval-depth must be >= 1")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    add_repo_to_path(args.repo)

    from benchmarks.chronos_bench import ChronosLiteBenchmarkAdapter  # noqa
    from run_benchmark import load_forecaster                          # noqa

    args.out.mkdir(parents=True, exist_ok=True)
    meta = corpus_meta(args.data)
    yaml_path = write_benchmark_yaml(args.data, args.out / "tasks.yaml")
    logger.info("12-task yaml -> %s", yaml_path)

    forecaster = load_forecaster(SimpleNamespace(
        model_path=args.model_path, device="cuda", torch_dtype="float32",
        config_path=None, model_source_path=None, batch_size=args.batch_size,
        eo_max_n_variate=None, rolling_horizon=None, rolling_quantiles=None,
        coe_eval_depth=args.depth))
    eo = getattr(getattr(forecaster, "_pipeline", None), "model", None)
    cfg = getattr(eo, "eo_config", None)
    if cfg is None:
        raise SystemExit(f"{args.model_path} is not an EO model")
    # NOT `coe_train_depth_max`. A checkpoint trained at K=1 with a grad-free
    # warm-up of 1 has been trained on an input one pass already refined, so
    # depth 2 is a depth it has seen; `checkpoint_depths` folds that in and four
    # sibling experiments already depend on it. Using K alone here would put the
    # `beyond_trained_depth` flag — and the red line in the figure — one column
    # too far left on such a checkpoint.
    trained, _, _ = checkpoint_depths(str(args.model_path))
    trained = int(trained or getattr(cfg, "coe_train_depth_max", 1))
    ctx_len = int(getattr(cfg, "context_length", 0))
    patch = int(getattr(cfg, "patch_size", 0) or 0)
    logger.info("checkpoint: trained_depth=%d context_length=%d patch=%d; "
                "scoring depths 1..%d", trained, ctx_len, patch, args.depth)
    if args.depth > trained:
        logger.info("depths %d..%d are DEEPER than this checkpoint was trained "
                    "for — recorded, and marked in the table",
                    trained + 1, args.depth)

    # The confound this corpus removes only stays removed if the model's window
    # is the one it was cut for. Reported, not enforced.
    report = observed_report(int(meta["observed_length"]), ctx_len, patch)
    for row in report:
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
        "model_path": str(args.model_path), "coe_eval_depth": args.depth,
        "trained_depth": trained, "context_length": ctx_len,
        "patch_size": patch, "data": str(args.data),
        "observed_length": int(meta["observed_length"]),
        "observed_report": report,
    }, indent=1))
    logger.info("wrote %s (%d rows)", dest, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
