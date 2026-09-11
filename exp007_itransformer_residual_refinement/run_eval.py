#!/usr/bin/env python3
"""Score an exp007 checkpoint on its dataset's test split.

Reports MSE and MAE in the paper's protocol — the whole test split, every
window, every variate, in the scaled space the model is trained in — and, for a
checkpoint with the refinement, the same pair at each chain depth so the
test-time-scaling curve comes out of ONE forward per batch.

Called directly by `train.py` when a run finishes, so a training job always
leaves a score behind, and standalone by `eval.sh` for a checkpoint that
already exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# CPU cap BEFORE numpy/torch: OpenMP and BLAS size their pools at load time.
# See finar_cpu.py.
sys.path.append(str(next(d for d in Path(__file__).resolve().parents
                         if (d / "finar_cpu.py").is_file())))
from finar_cpu import argv_workers, limit_cpu  # noqa: E402

limit_cpu(argv_workers(), quiet=True)

import torch  # noqa: E402

import paper  # noqa: E402
from data import build_dataset, build_loader  # noqa: E402
from itransformer import ITransformer, ModelConfig  # noqa: E402

logger = logging.getLogger("finar_exp007")


def load_checkpoint(run_dir: str | Path, device: str = "cpu"):
    """`(model, config, run metadata)` from a directory `train.py` wrote."""
    run_dir = Path(run_dir)
    ckpt = run_dir / "checkpoint.pt"
    conf = run_dir / "config.json"
    for path in (ckpt, conf):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing — {run_dir} is not an exp007 run "
                f"directory (train.py writes checkpoint.pt and config.json).")
    meta = json.loads(conf.read_text())
    cfg = ModelConfig.from_dict(meta["model"])
    model = ITransformer(cfg)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval(), cfg, meta


def resolve_depths(cfg: ModelConfig, depths) -> list[int]:
    """Which chain depths to score.

    A baseline checkpoint has exactly one: its embedding carries no forecast
    slot, so asking for depth 3 would silently score depth 1 three times.
    """
    if not cfg.coe_enabled:
        if depths and any(int(d) > 1 for d in depths):
            raise ValueError(
                "this checkpoint is a baseline (coe_enabled=false) — it has no "
                "forecast slot to refine, so --depths above 1 cannot be run. "
                "Train with --refinement on to sweep depth.")
        return [1]
    if depths:
        return sorted({max(1, int(d)) for d in depths})
    return [cfg.coe_eval_depth]


@torch.no_grad()
def test_metrics(model, cfg: ModelConfig, dataset, device: str,
                 batch_size: int, workers: int, depths: list[int]) -> dict:
    """MSE/MAE per depth, streamed.

    Streamed rather than stacked: the official `test()` keeps every prediction
    and every truth in memory before averaging, which on Traffic/720 is a
    17 GB pair of float32 arrays. MSE and MAE are means over all elements, so
    running sums give the identical number at constant memory. The sums are
    float64; the official mean is over float32, a difference far below the
    third decimal the paper reports.
    """
    loader = build_loader(dataset, "test", batch_size, workers)
    max_depth = max(depths)
    se = {d: 0.0 for d in depths}
    ae = {d: 0.0 for d in depths}
    count = 0
    model.eval()
    for batch_x, batch_y, x_mark, y_mark in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        if cfg.n_marks:
            x_mark = x_mark.float().to(device)
            y_mark = y_mark.float().to(device)
        else:
            x_mark = y_mark = None
        # ONE forward for the whole sweep: pass i's forecast is the input to
        # pass i+1, so the depths are already nested inside a single chain.
        outs = model(batch_x, x_mark, y_mark, depth=max_depth,
                     return_all_passes=True)
        count += batch_y.numel()
        for d in depths:
            diff = (outs[d - 1] - batch_y).double()
            se[d] += float((diff * diff).sum())
            ae[d] += float(diff.abs().sum())
    if not count:
        raise RuntimeError("the test split produced no windows")
    return {d: {"mse": se[d] / count, "mae": ae[d] / count} for d in depths}


def compare_to_paper(dataset: str, pred_len: int, mse: float, mae: float):
    """`(verdict, detail)` against Table 10, or `("no-target", ...)`."""
    want = paper.target(dataset, pred_len)
    if want is None:
        return "no-target", f"{dataset}/{pred_len} is not in the paper table"
    ok_mse = paper.within_tolerance(mse, want[0])
    ok_mae = paper.within_tolerance(mae, want[1])
    detail = (f"paper MSE {want[0]:.3f} MAE {want[1]:.3f} | "
              f"got MSE {mse:.4f} ({mse - want[0]:+.4f}, "
              f"{100 * (mse - want[0]) / want[0]:+.1f}%) "
              f"MAE {mae:.4f} ({mae - want[1]:+.4f}, "
              f"{100 * (mae - want[1]) / want[1]:+.1f}%)")
    return ("match" if ok_mse and ok_mae else "miss"), detail


def evaluate(run_dir: str | Path, data_root: str, device: str,
             batch_size: int, workers: int, depths=None) -> dict:
    """Score one run directory and write `metrics.json` into it."""
    run_dir = Path(run_dir)
    model, cfg, meta = load_checkpoint(run_dir, device)
    depths = resolve_depths(cfg, depths)
    dataset = build_dataset(meta["dataset"], data_root, "test",
                            cfg.seq_len, cfg.pred_len, meta.get("freq", "h"))
    scores = test_metrics(model, cfg, dataset, device, batch_size, workers,
                          depths)
    # The HEADLINE number is the depth the checkpoint is configured to run, not
    # the deepest one a sweep happened to include: `--depths 1 2 3 4` on a
    # checkpoint whose coe_eval_depth is 3 is a sweep AROUND its setting, and
    # quietly reporting depth 4 would let the flag change the result.
    headline = cfg.coe_eval_depth if cfg.coe_eval_depth in scores else max(depths)
    best = scores[headline]
    verdict, detail = compare_to_paper(
        meta["dataset"], cfg.pred_len, best["mse"], best["mae"])
    if cfg.coe_enabled and verdict != "no-target":
        # The published cell is the PAPER'S architecture. For a refined
        # checkpoint this comparison is informative, not a reproduction claim.
        detail += " [refined variant against the paper's baseline cell]"
    out = {
        "run": run_dir.name,
        "dataset": meta["dataset"],
        "variant": cfg.variant,
        "seq_len": cfg.seq_len,
        "pred_len": cfg.pred_len,
        "test_windows": len(dataset),
        "by_depth": {str(d): scores[d] for d in depths},
        "headline_depth": headline,
        "mse": best["mse"],
        "mae": best["mae"],
        "paper_verdict": verdict,
        "paper_detail": detail,
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    for d in depths:
        logger.info("%s depth=%d  MSE %.4f  MAE %.4f", run_dir.name, d,
                    scores[d]["mse"], scores[d]["mae"])
    logger.info("%s vs paper (depth %d): %s — %s", run_dir.name, headline,
                verdict.upper(), detail)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Score an exp007 checkpoint on the test split.")
    p.add_argument("--run-dir", required=True,
                   help="a directory train.py wrote (checkpoint.pt + "
                        "config.json)")
    p.add_argument("--data-root", required=True,
                   help="root holding ETT-small/, electricity/, ... "
                        "(see prepare_data.sh)")
    p.add_argument("--depths", nargs="+", default=None,
                   help="chain depths to score in one pass, e.g. 1 2 3 4. "
                        "Default: the checkpoint's coe_eval_depth.")
    p.add_argument("--batch-size", type=int, default=32,
                   help="eval batch size. The official loader uses 1; MSE/MAE "
                        "are means over every window either way (default: 32)")
    p.add_argument("--num-workers", type=int, default=None,
                   help="CPU cap for this run. Default: derived from the "
                        "scheduler affinity mask and the cgroup quota. A value "
                        "above the allocation is clamped, not obeyed.")
    p.add_argument("--loader-workers", type=int, default=2,
                   help="DataLoader worker processes, within the CPU cap "
                        "(default: 2)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")
    cap = limit_cpu(args.num_workers)
    evaluate(args.run_dir, args.data_root, args.device, args.batch_size,
             min(args.loader_workers, cap), args.depths)


if __name__ == "__main__":
    main()
