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

THE HORIZON IS A SLICE, NOT A CORPUS
The grid is built once at length 2048 (`build_dataset.py`), and a horizon is a
VIEW of it. Two things follow, and both are the point:

  * reading a trend down the H axis compares one corpus with itself, not four
    independent draws — which is what the previous build did, and what made a
    per-horizon trend uninterpretable;
  * every horizon is scored on the SAME context window. The window is anchored
    at ``2048 - 1024`` so it does not move when `--horizons` selects a subset,
    and it ends where the longest horizon begins.

Fixing the context is not cosmetic. With ``offset = -H`` the evaluator's
context is everything before the split, so it would otherwise be ``2048 - H``
and shrink by a factor of 64 across this sweep — the H axis would then vary
horizon length and context length together.

WHY THE VIEWS ARE MATERIALISED, given three cheaper-looking alternatives:

  * `series_end` in the yaml CAN pin the context, but only at 1024.
    `series_trim` sets ``min_len = |offset| + prediction_length = 2H``, so an
    entry trimmed to ``C + H`` survives only while ``C + H >= 2H``, i.e.
    ``C >= H``. At C = 1024 that holds for the whole sweep and would cost
    nothing; at C = 512 it drops every series at H > 512. The context has to be
    512 for the reason below, so this route is closed by the value, not by the
    mechanism.
  * capping the history in the forecaster instead (the shape `time_bench.py`
    uses via `max_context`) would fix what the MODEL sees but not the MASE
    scale, which `gluonts_split` computes from the whole past segment. That
    denominator would still be ``2048 - H`` and still vary down the H axis,
    reintroducing the confound the fixed context exists to remove.
  * the general fix is a scalar `context_length` field on a benchmark entry,
    instead of the per-item `series_end` dict. That lives in tsm-trainer's
    evaluator, which this experiment imports and must not modify.

Distinct directory names then come for free, which the collision would
otherwise have needed on its own: the evaluator names a result row after the
dataset, so four horizons over one name would produce four indistinguishable
rows.

WHY 512 STEPS OF CONTEXT
An oracle that grid-searches the oscillator period on the context and
extrapolates reaches horizon R^2 of 0.96 / 0.96 / 0.95 / 0.85 at H = 16 / 64 /
256 / 1024 from 512 steps, and -7 / -944 at the two long horizons from 64
steps (negative = worse than predicting the mean, because a frequency error
estimated from a short context accumulates linearly into a phase error). 512 is
the shortest context at which all four horizons are measurable at all.

It is also the reason the views are materialised rather than expressed as
`series_end` entries, which would pin the context at 1024: by 1024 steps the
same oracle is near 0.96 at every horizon, so the difficulty gradient across H
— the thing the horizon axis exists to measure — would be flattened away.

Usage
-----
    python run_eval.py --model-path <ckpt> --data-root <corpora> --out <dir>
    python run_eval.py ... --coe-eval-depth 2       # default 4
    python run_eval.py ... --context 256            # default 512
    python run_eval.py ... --horizons 16 64         # subset of the sweep
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

import numpy as np
import yaml

logger = logging.getLogger("finar_exp001_eval")

#: The repo is imported, never modified. Overridable for a different checkout.
DEFAULT_REPO = Path("/group-volume/workspace/mun-hak.lee/experiments/"
                    "tsm-trainer_001/tsm-trainer")

#: The evaluation protocol, IMPORTED from the module that builds the corpus
#: rather than restated here. `build_dataset.py` sizes the series for this
#: sweep, so a `--length` or a fourth horizon that moved only one of the two
#: copies would leave this file slicing a valid corpus at the wrong offsets —
#: with no error, since every view would still be well formed.
from build_dataset import (  # noqa: E402  (a sibling module, same directory)
    EVAL_CONTEXT as CONTEXT,
    EVAL_HORIZONS as HORIZONS,
    SERIES_LENGTH,
    target_features,
)

#: Where every horizon starts. Derived from the LONGEST horizon, so selecting a
#: subset with `--horizons` leaves each context window exactly where it was.
ANCHOR = SERIES_LENGTH - max(HORIZONS)


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


def view_name(shard: str, context: int, H: int) -> str:
    return f"{shard}_c{context}h{H}"


def make_views(data_root: Path, views_root: Path, shards: list[str],
               context: int, horizons, force: bool = False,
               anchor: int = ANCHOR) -> list[tuple[str, int]]:
    """Materialise ``context + H`` slices of every shard, one per horizon.

    Returns the ``[(view name, H)]`` the yaml should list.

    A view is ``target[..., ANCHOR - context : ANCHOR + H]``, so the context
    window is byte-identical across the four horizons and only the horizon
    grows. They are real corpora rather than symlinks because they differ in
    length — but they are DETERMINISTIC SLICES OF ONE BUILD, so the objection
    that killed the previous design (four independently generated corpora)
    does not apply.

    Distinct directory names matter: the evaluator names a result row after the
    dataset, so four views of one shard under one name would collide and the
    pairwise tools reject a csv with duplicate keys.
    """
    import datasets as hf

    views_root.mkdir(parents=True, exist_ok=True)
    lo = anchor - context
    if lo < 0:
        # `anchor`, not `max(horizons)`: the anchor is set by the LONGEST
        # horizon of the sweep, so quoting the selected subset here would
        # describe a relationship that does not hold under `--horizons`.
        raise SystemExit(
            f"--context {context} does not fit before the horizon anchor at "
            f"{anchor}, which leaves {SERIES_LENGTH - anchor} steps for the "
            f"longest horizon of the sweep")
    made, written = [], 0
    for shard in shards:
        src = None
        for H in horizons:
            name = view_name(shard, context, H)
            dest = views_root / name
            made.append((name, H))
            if dest.is_dir() and not force:
                continue
            written += 1
            if src is None:
                ds = hf.load_from_disk(str(data_root / shard))
                src = ds["train"] if hasattr(ds, "keys") else ds
                # `with_format("numpy")` decodes arrow in C. Reading the column
                # in the default format materialises 2.1M Python floats per
                # shard — 67 MB of objects, 216 times — to build an array that
                # arrow can hand over directly.
                tgt = np.asarray(src.with_format("numpy")["target"],
                                 dtype=np.float32)
                cols = {c: src[c] for c in ("item_id", "start", "freq")
                        if c in src.column_names}
            cut = tgt[..., lo:anchor + H]
            hf.DatasetDict({"train": hf.Dataset.from_dict(
                {**cols, "target": [x.tolist() for x in cut]},
                features=target_features())}).save_to_disk(str(dest))
    # Says how many were WRITTEN, not just how many exist: the old message
    # quoted the total either way, so a run that reused every slice looked
    # exactly like one that regenerated the corpus.
    logger.info("views: %d of %d slices written (%d reused) under %s "
                "— context %d, window [%d, %d) + horizon",
                written, len(made), len(made) - written, views_root,
                context, lo, anchor)
    return made


def write_config(data_root: Path, cells: list[tuple[str, int]],
                 out: Path) -> Path:
    """A chronos-style benchmark yaml for the selected views.

    ``offset = -prediction_length`` with ``num_rolls = 1`` scores exactly the
    last H steps of each view once. Since a view is exactly ``context + H``
    long, that leaves precisely ``context`` steps of history — which is the
    whole reason the views exist.
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
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS),
                   help="horizons to slice and score (default: 16 64 256 1024)")
    p.add_argument("--context", type=int, default=CONTEXT,
                   help="observed steps before every horizon. Held EQUAL "
                        "across horizons, which is the point of the views; "
                        "see WHY 512 STEPS OF CONTEXT in the module docstring")
    p.add_argument("--views-root", type=Path, default=None,
                   help="where the context+H slices go (default: "
                        "<data-root>/views)")
    p.add_argument("--force-views", action="store_true",
                   help="rewrite the slices even if they already exist")
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
    shards = [d.name for d in sorted(args.data_root.iterdir())
              if d.name.startswith("phi")
              and ((d / "dataset_dict.json").is_file() or (d / "train").is_dir())]
    if not shards:
        raise SystemExit(
            f"no corpus shards under {args.data_root} (expected directories "
            f"named phi<level>_sh<level>_gp<level>_s<k>)")

    views_root = args.views_root or (args.data_root / "views")
    cells = make_views(args.data_root, views_root, shards, args.context,
                       args.horizons, force=args.force_views)
    logger.info("scoring %d shards x %d horizons at depths 1..%d",
                len(shards), len(args.horizons), args.coe_eval_depth)

    args.out.mkdir(parents=True, exist_ok=True)
    cfg_path = write_config(views_root, cells, args.out / "benchmark.yaml")
    (args.out / "protocol.json").write_text(json.dumps({
        "context": args.context, "horizons": list(args.horizons),
        "anchor": ANCHOR, "series_length": SERIES_LENGTH,
        "context_window": [ANCHOR - args.context, ANCHOR],
        "data_root": str(args.data_root), "views_root": str(views_root),
        "coe_eval_depth": args.coe_eval_depth,
    }, indent=1))

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
