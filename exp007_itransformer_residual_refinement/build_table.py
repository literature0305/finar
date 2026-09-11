#!/usr/bin/env python3
"""Collect exp007's runs into one table and one figure.

Reads every `metrics.json` under a results root and answers the experiment's
question directly: for each (dataset, horizon), what did the baseline score,
what did the refinement score, and did the refinement help. The published cell
is carried alongside, so a comparison against a baseline that did NOT reproduce
the paper is visible as such rather than quietly presented as a result.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(next(d for d in Path(__file__).resolve().parents
                         if (d / "finar_cpu.py").is_file())))
from finar_cpu import limit_cpu  # noqa: E402

limit_cpu(quiet=True)

import paper  # noqa: E402

logger = logging.getLogger("finar_exp007")

FIELDS = ("dataset", "pred_len", "variant", "depth", "headline", "mse",
          "mae", "paper_mse", "paper_mae", "verdict", "run")


def collect(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        want = paper.target(m["dataset"], m["pred_len"]) or (None, None)
        for depth, score in sorted(m["by_depth"].items(), key=lambda kv: int(kv[0])):
            rows.append({
                "dataset": m["dataset"], "pred_len": m["pred_len"],
                "variant": m["variant"], "depth": int(depth),
                "mse": round(score["mse"], 6), "mae": round(score["mae"], 6),
                "paper_mse": want[0], "paper_mae": want[1],
                # The verdict belongs to the run's HEADLINE depth — the one
                # the checkpoint is configured to run. A sweep's other depths
                # are measurements around it, not what it claims to be.
                "verdict": (m["paper_verdict"]
                            if int(depth) == int(m.get(
                                "headline_depth",
                                max(int(d) for d in m["by_depth"])))
                            else ""),
                "headline": int(depth) == int(m.get(
                    "headline_depth", max(int(d) for d in m["by_depth"]))),
                "run": m["run"],
            })
    return rows


def compare(rows: list[dict]) -> list[dict]:
    """Baseline vs each refinement variant, per (dataset, horizon).

    TWO gains are reported, because they answer different questions and only
    one of them is a result:

      `improvement_pct` — at the depth the checkpoint is CONFIGURED to run.
        This is the honest number: the depth was chosen before the test split
        was scored.
      `improvement_pct_best_depth` — at whichever swept depth scored best.
        That is selection on the test set, so it is an upper bound on what
        picking a depth could buy, not a score. It is reported because a gain
        that exists at only one depth should be visible as such, and labelled
        so it cannot be quoted as the result.
    """
    headline, best_at = {}, {}
    for r in rows:
        key = (r["dataset"], r["pred_len"], r["variant"])
        if r.get("headline"):
            headline[key] = r
        if key not in best_at or r["mse"] < best_at[key]["mse"]:
            best_at[key] = r
    out = []
    for key in sorted(headline):
        dataset, horizon, variant = key
        if variant == "baseline":
            continue
        row, best = headline[key], best_at[key]
        base = headline.get((dataset, horizon, "baseline"))
        if base is None:
            out.append({"dataset": dataset, "pred_len": horizon,
                        "variant": variant, "baseline_mse": None,
                        "refined_mse": row["mse"], "improvement_pct": None,
                        "eval_depth": row["depth"],
                        "refined_mse_best_depth": best["mse"],
                        "improvement_pct_best_depth": None,
                        "best_depth": best["depth"],
                        "note": "no baseline run to compare against"})
            continue
        gain = 100.0 * (base["mse"] - row["mse"]) / base["mse"]
        gain_best = 100.0 * (base["mse"] - best["mse"]) / base["mse"]
        note = ""
        if base["verdict"] == "miss":
            note = ("the baseline missed its published cell — read the gain "
                    "as relative to THIS baseline, not to the paper's")
        out.append({
            "dataset": dataset, "pred_len": horizon, "variant": variant,
            "baseline_mse": round(base["mse"], 6),
            "refined_mse": round(row["mse"], 6),
            "improvement_pct": round(gain, 3),
            "eval_depth": row["depth"],
            "refined_mse_best_depth": round(best["mse"], 6),
            "improvement_pct_best_depth": round(gain_best, 3),
            "best_depth": best["depth"], "note": note,
        })
    return out


def figure(rows: list[dict], comparisons: list[dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed — table only, no figure")
        return False
    if not comparisons:
        logger.info("no baseline/refinement pair yet — no figure")
        return False

    depth_rows = [r for r in rows if r["variant"] != "baseline"]
    sweeps = {}
    for r in depth_rows:
        sweeps.setdefault((r["dataset"], r["pred_len"], r["variant"]), []).append(r)
    sweeps = {k: sorted(v, key=lambda r: r["depth"])
              for k, v in sweeps.items() if len(v) > 1}

    ncols = 2 if sweeps else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4.5), squeeze=False)
    ax = axes[0][0]
    labels = [f"{c['dataset']}\n{c['pred_len']}" for c in comparisons]
    gains = [c["improvement_pct"] or 0.0 for c in comparisons]
    colors = ["#2b7a3d" if g > 0 else "#a33" for g in gains]
    ax.bar(range(len(gains)), gains, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MSE improvement over baseline (%)")
    ax.set_title("exp007: residual refinement vs the paper's iTransformer")

    if sweeps:
        ax = axes[0][1]
        for (dataset, horizon, variant), rs in sorted(sweeps.items()):
            base = next((c["baseline_mse"] for c in comparisons
                         if c["dataset"] == dataset
                         and c["pred_len"] == horizon), None)
            ys = [100.0 * (base - r["mse"]) / base if base else r["mse"]
                  for r in rs]
            ax.plot([r["depth"] for r in rs], ys, marker="o",
                    label=f"{dataset}/{horizon} {variant}")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("chain depth at inference (coe_eval_depth)")
        ax.set_ylabel("MSE improvement over baseline (%)")
        ax.set_title("test-time scaling")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", help="directory holding the run directories")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        datefmt="%m-%d %H:%M:%S")
    root = Path(args.root)
    rows = collect(root)
    if not rows:
        logger.warning("no metrics.json under %s — nothing to table", root)
        return
    table = root / "exp007_table.csv"
    with open(table, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("%d row(s) -> %s", len(rows), table)

    comparisons = compare(rows)
    if comparisons:
        path = root / "exp007_improvement.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        logger.info("%d comparison(s) -> %s", len(comparisons), path)
        for c in comparisons:
            logger.info("  %s/%s %s: baseline %s -> %s at depth %s (%s%%); "
                        "best swept depth %s gives %s (%s%%, test-set "
                        "selection) %s",
                        c["dataset"], c["pred_len"], c["variant"],
                        c["baseline_mse"], c["refined_mse"], c["eval_depth"],
                        c["improvement_pct"], c["best_depth"],
                        c["refined_mse_best_depth"],
                        c["improvement_pct_best_depth"], c["note"])
    if figure(rows, comparisons, root / "exp007_refinement.png"):
        logger.info("figure -> %s", root / "exp007_refinement.png")

    missed = sorted({f"{r['dataset']}/{r['pred_len']}" for r in rows
                     if r["variant"] == "baseline" and r["verdict"] == "miss"})
    if missed:
        logger.warning("baseline(s) outside the published tolerance: %s — the "
                       "refinement's gain is measured against these, so say so "
                       "when reporting it", ", ".join(missed))


if __name__ == "__main__":
    main()
