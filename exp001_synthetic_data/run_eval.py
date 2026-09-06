#!/usr/bin/env python3
"""Score one model on the FiNAR exp001 grid, at every recursion depth.

WHY THIS IMPORTS tsm-trainer INSTEAD OF CALLING run_evaluation.sh
`run_benchmark.py` resolves every benchmark's yaml from a hard-coded
`configs_dir = SCRIPT_DIR / "configs"`, so the only way to evaluate a corpus
that lives outside the repo through that entry point is to drop a yaml INSIDE
the repo — which the work order forbids. `Evaluator.evaluate_benchmark()`
takes an arbitrary `config_path`, so this driver writes the yaml here, in the
finar repo, and calls the evaluator directly. tsm-trainer is imported and never
written to.

WHAT COMES OUT
The evaluator's generic yaml path already emits per-recursion-depth columns —
`score_and_plot_repeats` writes `repeat<d>_MASE` / `repeat<d>_WQL` per task
whenever the forecaster reports more than one depth. Those columns are what
`build_table.py` reads for the iteration-1 vs iteration-2 comparison; nothing
here recomputes them.

DEPTH GUARD
A checkpoint trained with `coe_train_depth_max < 2` has no second pass to
report, and asking for one measures extrapolation past the training regime
rather than refinement. Worse, it fails SILENTLY: `_repeats_worth_asking`
returns False as soon as `report_depth() == 1`, before it ever reads
`TSM_FEV_COE_REPEATS`, and the log line that would explain the absence is
itself gated on having depths to report. This driver therefore reads the
checkpoint's config and refuses up front.

Usage
-----
    python run_eval.py --model-path <ckpt> --data-root <corpora> --out <dir>
    python run_eval.py ... --coe-eval-depth 2       # default 4
    python run_eval.py ... --horizons 16 128        # subset of the grid
    python run_eval.py --model-path autogluon/chronos-2 ...   # baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

logger = logging.getLogger("finar_exp001_eval")

#: The repo is imported, never modified. Overridable for a different checkout.
DEFAULT_REPO = Path("/group-volume/workspace/mun-hak.lee/experiments/"
                    "tsm-trainer_001/tsm-trainer")


def add_repo_to_path(repo: Path) -> None:
    ev = repo / "scripts" / "forecasting" / "evaluation"
    if not ev.is_dir():
        raise SystemExit(f"no evaluation package under {repo} (looked at {ev})")
    for p in (str(ev), str(repo / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def checkpoint_depths(model_path: str) -> tuple[int | None, int | None]:
    """``(coe_train_depth_max, coe_eval_depth)`` from a checkpoint's config.

    Returns ``(None, None)`` for anything that is not a local EO v4 checkpoint
    — an HF id like `autogluon/chronos-2` has no eo_config and is a legitimate
    baseline, so the guard has to distinguish "not a COE model" from "a COE
    model with nothing to sweep".
    """
    cfg = Path(model_path) / "config.json"
    if not cfg.is_file():
        return None, None
    try:
        eo = json.loads(cfg.read_text()).get("eo_config") or {}
    except Exception:
        return None, None
    return eo.get("coe_train_depth_max"), eo.get("coe_eval_depth")


def write_config(data_root: Path, cells: list[tuple[str, int]],
                 out: Path) -> Path:
    """A chronos-style benchmark yaml for the selected cells.

    ``offset = -prediction_length`` with ``num_rolls = 1`` scores exactly the
    last H steps of each series once, which is what the corpus was built for:
    the generator appends H horizon steps after C context steps, so any other
    offset would score part of the context.
    """
    cfg = {
        "datasets_root": str(data_root),
        "datasets": [{"name": name, "offset": -H, "prediction_length": H,
                      "num_rolls": 1} for name, H in cells],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True,
                   help="EO v4 checkpoint dir, or an HF id for a baseline")
    p.add_argument("--data-root", type=Path,
                   default=Path("/group-volume/ts-dataset/finar_exp001"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--coe-eval-depth", type=int, default=4,
                   help="recursion depth to score at. Default 4; the sweep "
                        "reports every depth from 1 to this. May exceed the "
                        "checkpoint's training depth on purpose — that is the "
                        "extrapolation arm — but never go below 2.")
    p.add_argument("--horizons", type=int, nargs="+", default=None,
                   help="subset of the grid's horizons (default: all present)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--allow-shallow", action="store_true",
                   help="run a non-COE or depth-1 model anyway, for the "
                        "Chronos-2 baseline that has no iteration axis")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    if args.coe_eval_depth < 2 and not args.allow_shallow:
        raise SystemExit(
            f"--coe-eval-depth {args.coe_eval_depth} would report a single "
            f"depth, and the whole experiment is an iteration-1 vs iteration-2 "
            f"comparison. Pass >= 2, or --allow-shallow for a baseline.")

    train_max, ckpt_eval = checkpoint_depths(args.model_path)
    if train_max is None:
        if not args.allow_shallow:
            raise SystemExit(
                f"{args.model_path} has no eo_config — it is not an EO v4 "
                f"checkpoint, so it has no recursion depth to sweep. Pass "
                f"--allow-shallow to score it as a baseline.")
        logger.warning("no eo_config at %s — scoring as a non-COE baseline",
                       args.model_path)
    else:
        logger.info("checkpoint depths: coe_train_depth_max=%s coe_eval_depth=%s",
                    train_max, ckpt_eval)
        if train_max < 2 and not args.allow_shallow:
            raise SystemExit(
                f"coe_train_depth_max={train_max} < 2: this checkpoint was "
                f"trained without iterative refinement, so iteration 2 is not "
                f"a refinement of iteration 1 but an extrapolation past the "
                f"training regime. Use a K>=2 checkpoint, or --allow-shallow "
                f"to record that arm deliberately.")
        if args.coe_eval_depth > train_max:
            logger.warning(
                "--coe-eval-depth %d exceeds the checkpoint's "
                "coe_train_depth_max=%d — this is the extrapolation arm; "
                "depths above %d are outside the trained range",
                args.coe_eval_depth, train_max, train_max)

    # Discover the grid from disk, so a partially built corpus is obvious here
    # rather than as a missing row in the final table.
    if not args.data_root.is_dir():
        raise SystemExit(
            f"--data-root {args.data_root} does not exist. Build the grid "
            f"first:\n  python build_dataset.py --out-root {args.data_root}")
    cells = []
    for d in sorted(args.data_root.iterdir()):
        if not (d / "dataset_dict.json").is_file() and not (d / "train").is_dir():
            continue
        if not d.name.startswith("H"):
            continue
        H = int(d.name.split("_")[0][1:])
        if args.horizons and H not in args.horizons:
            continue
        cells.append((d.name, H))
    if not cells:
        raise SystemExit(f"no corpus cells under {args.data_root}")
    logger.info("scoring %d cells at depths 1..%d", len(cells),
                args.coe_eval_depth)

    args.out.mkdir(parents=True, exist_ok=True)
    cfg_path = write_config(args.data_root, cells, args.out / "benchmark.yaml")

    add_repo_to_path(args.repo)
    # Belt and braces: the fev adapter gates its depth sweep on this, and a
    # future adapter may too. Harmless where report_depth already forces it.
    os.environ.setdefault("TSM_FEV_COE_REPEATS", "1")

    # run_benchmark.load_forecaster is the factory, reused rather than
    # reimplemented: it is what decides EOForecaster vs Chronos2Forecaster from
    # the checkpoint, and a second copy of that rule here would be the thing
    # that goes stale when a new model type lands. It reads a namespace, so it
    # gets one.
    from types import SimpleNamespace  # noqa: E402  (needs sys.path above)

    from engine.evaluator import Evaluator  # noqa: E402
    from run_benchmark import load_forecaster  # noqa: E402

    t0 = time.time()
    forecaster = load_forecaster(SimpleNamespace(
        model_path=args.model_path,
        device="cuda",
        torch_dtype="float32",
        config_path=None,
        model_source_path=None,
        batch_size=args.batch_size,
        eo_max_n_variate=None,
        rolling_horizon=None,
        rolling_quantiles=None,
        # None leaves the checkpoint in charge, which is what a non-COE
        # baseline needs; the guards above already refused the cases where a
        # depth is required and absent.
        coe_eval_depth=(None if train_max is None else args.coe_eval_depth),
    ))
    ev = Evaluator(forecaster, batch_size=args.batch_size)
    df = ev.evaluate_benchmark(
        config_path=cfg_path,
        output_path=args.out / "finar_exp001.csv",
        benchmark_name="finar_exp001",
        output_dir=args.out,
    )
    logger.info("scored %d rows in %.1fs -> %s", len(df), time.time() - t0,
                args.out / "finar_exp001.csv")
    depth_cols = [c for c in df.columns if c.startswith("repeat")]
    if depth_cols:
        logger.info("per-depth columns: %s", sorted(depth_cols))
    else:
        logger.warning(
            "NO repeat<d> columns — the run reported a single depth. Check "
            "the checkpoint's coe_train_depth_max and --coe-eval-depth; "
            "build_table.py will have no iteration axis to compare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
