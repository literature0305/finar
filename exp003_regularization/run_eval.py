#!/usr/bin/env python3
"""Score a model on the exp003 5k training subset, at three horizons.

exp003 asks whether the second recursion pass is buying dependency modelling
or acting as a regulariser. The distinguishing evidence is how the iteration-1
to iteration-2 gain behaves on data drawn the way TRAINING draws it, across a
horizon sweep: a regulariser should help roughly uniformly, while dependency
modelling should help more as the horizon lengthens and there is more joint
structure to get wrong in one pass.

Reuses exp001's approach rather than reimplementing it. `run_benchmark.py`
resolves benchmark configs from a hard-coded directory inside tsm-trainer, so
routing an external corpus through it would mean writing a file there;
`Evaluator.evaluate_benchmark()` takes an arbitrary `config_path`, so the yaml
lives here and tsm-trainer is imported, never modified.

Every recursion depth is reported: the generic yaml path emits
`repeat<d>_MASE` / `repeat<d>_WQL` per dataset whenever the forecaster has more
than one depth, which is what makes the iteration comparison possible.

Usage
-----
    python run_eval.py --model-path <ckpt> --repo <tsm-trainer> --out <dir>
    python run_eval.py ... --coe-eval-depth 2       # default 2
    python run_eval.py --model-path autogluon/chronos-2 ... --allow-shallow
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("finar_exp003")

DEFAULT_REPO = Path("/group-volume/workspace/mun-hak.lee/experiments/"
                    "tsm-trainer_001/tsm-trainer")
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_subset_5k.yaml"


def add_repo_to_path(repo: Path) -> None:
    ev = repo / "scripts" / "forecasting" / "evaluation"
    if not ev.is_dir():
        raise SystemExit(f"no evaluation package under {repo} (looked at {ev})")
    for p in (str(ev), str(repo / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def checkpoint_depths(model_path: str) -> tuple[int | None, int | None]:
    """``(coe_train_depth_max, coe_eval_depth)``, or ``(None, None)``.

    None means "not a local EO v4 checkpoint" — an HF id like
    `autogluon/chronos-2` is a legitimate baseline here, so this has to
    distinguish that from a COE model with nothing to sweep.
    """
    cfg = Path(model_path) / "config.json"
    if not cfg.is_file():
        return None, None
    try:
        eo = json.loads(cfg.read_text()).get("eo_config") or {}
    except Exception:
        return None, None
    return eo.get("coe_train_depth_max"), eo.get("coe_eval_depth")


def check_corpus(config_path: Path) -> None:
    """Fail early when the corpus or its horizon links are absent.

    The symlinks are what give three horizons three distinguishable names, and
    a missing one surfaces deep inside the loader as a confusing path error.
    """
    import yaml

    cfg = yaml.safe_load(config_path.read_text())
    root = Path(cfg["datasets_root"])
    missing = [d["name"] for d in cfg["datasets"] if not (root / d["name"]).exists()]
    if missing:
        raise SystemExit(
            f"{missing} not under {root}. Build the corpus first:\n"
            f"  python build_subset.py --config <training yaml> --repo <tsm-trainer>")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True,
                   help="EO v4 checkpoint dir, or an HF id for a baseline")
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO,
                   help="tsm-trainer checkout to IMPORT from (read-only)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--coe-eval-depth", type=int, default=2,
                   help="depth to score at; every depth 1..N is reported")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--allow-shallow", action="store_true",
                   help="score a model with no iteration axis as a baseline")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    check_corpus(args.config)

    train_max, ckpt_eval = checkpoint_depths(args.model_path)
    if train_max is None:
        if not args.allow_shallow:
            raise SystemExit(
                f"{args.model_path} has no eo_config — no recursion depth to "
                f"sweep. Pass --allow-shallow to score it as a baseline.")
        logger.warning("no eo_config at %s — scoring as a non-COE baseline",
                       args.model_path)
    else:
        logger.info("checkpoint: coe_train_depth_max=%s coe_eval_depth=%s",
                    train_max, ckpt_eval)
        if train_max < 2 and not args.allow_shallow:
            raise SystemExit(
                f"coe_train_depth_max={train_max} < 2: this checkpoint was not "
                f"trained to refine, so iteration 2 would be extrapolation "
                f"rather than the refinement exp003 is analysing. Use a K>=2 "
                f"checkpoint, or --allow-shallow to record that arm anyway.")
        if args.coe_eval_depth > train_max:
            logger.warning("--coe-eval-depth %d exceeds coe_train_depth_max=%d "
                           "— depths above %d are outside the trained range",
                           args.coe_eval_depth, train_max, train_max)

    add_repo_to_path(args.repo)
    # The fev adapter gates its depth sweep on this; harmless elsewhere.
    os.environ["TSM_FEV_COE_REPEATS"] = "1"

    from types import SimpleNamespace  # noqa: E402

    from engine.evaluator import Evaluator  # noqa: E402
    from run_benchmark import load_forecaster  # noqa: E402

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    forecaster = load_forecaster(SimpleNamespace(
        model_path=args.model_path, device="cuda", torch_dtype="float32",
        config_path=None, model_source_path=None, batch_size=args.batch_size,
        eo_max_n_variate=None, rolling_horizon=None, rolling_quantiles=None,
        coe_eval_depth=(None if train_max is None else args.coe_eval_depth),
    ))
    ev = Evaluator(forecaster, batch_size=args.batch_size)
    df = ev.evaluate_benchmark(
        config_path=args.config,
        output_path=args.out / "train_subset_5k.csv",
        benchmark_name="train_subset_5k",
        output_dir=args.out,
    )
    logger.info("scored %d rows in %.1fs -> %s", len(df), time.time() - t0,
                args.out / "train_subset_5k.csv")
    depth_cols = sorted(c for c in df.columns if c.startswith("repeat"))
    if depth_cols:
        logger.info("per-depth columns: %s", depth_cols)
    else:
        logger.warning(
            "NO repeat<d> columns — the run reported a single depth, so there "
            "is no iteration axis to analyse. Check --coe-eval-depth and the "
            "checkpoint's coe_train_depth_max.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
