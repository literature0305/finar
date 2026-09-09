#!/usr/bin/env python3
"""Step 1 vs step 2 per model — table and figure.

Reads the per-model csvs `run_eval.py` leaves in the results directory and
reports, per model and benchmark, whether feeding a model its OWN covariate
forecasts as known-future values improves its target forecast:

    model            benchmark   n   step1 MASE  step2 MASE  improve%  win%

`improvement` is ``(step1 - step2) / step1``, so POSITIVE means the second,
pseudo-known-future pass helped. Both steps score variate 0 only, on the same
items, so the pair answers one question about one series.

WHAT A ZERO ROW WOULD MEAN. `n_identical` counts items where the two steps
returned the same forecast. A model that ignores `future_covariates` produces
an all-identical row, which reads as "no gain" but is really "the manipulation
never reached the model" — `run_eval.py` refuses a fully-identical run, and
this table carries the count so a partially-identical one is visible too.

Usage
-----
    python build_table.py <results-dir>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("finar_exp005_table")

METRICS = ("MASE", "WQL")


def load(root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(c) for c in sorted(root.glob("*.csv"))
              if not c.name.endswith("_table.csv")]
    if not frames:
        raise SystemExit(f"no per-model csv under {root}")
    return pd.concat(frames, ignore_index=True)


def build(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, benchmark, metric), pooled over tasks.

    Pooled by ITEM COUNT, not by task: a 32-item GIFT task and a 10-item fev
    window carry different evidence and a plain task mean would weight them
    equally. `n_scored` is the weight the metrics were computed on.
    """
    rows = []
    for (model, bench), g in df.groupby(["model", "benchmark"], dropna=False):
        w = g["n_scored"].to_numpy(float)
        for metric in METRICS:
            a = g[f"step1_{metric}"].to_numpy(float)
            b = g[f"step2_{metric}"].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b) & (w > 0)
            if not ok.any():
                continue
            s1 = float(np.average(a[ok], weights=w[ok]))
            s2 = float(np.average(b[ok], weights=w[ok]))
            # Per-task improvement, then weighted — a ratio of the pooled means
            # would let one large-scale task set the sign for the whole row.
            per_task = (a[ok] - b[ok]) / np.where(a[ok] == 0, np.nan, a[ok]) * 100
            rows.append({
                "model": model, "benchmark": bench, "metric": metric,
                "n_tasks": int(ok.sum()), "n_items": int(w[ok].sum()),
                f"step1_{metric}": s1, f"step2_{metric}": s2,
                "improvement_pct": float(np.nansum(per_task * w[ok])
                                         / np.nansum(w[ok])),
                "win_rate": float(np.average(
                    g["win_rate"].to_numpy(float)[ok], weights=w[ok])),
                "n_identical": int(g["n_identical"].to_numpy()[ok].sum()),
            })
    return pd.DataFrame(rows)


def render(t: pd.DataFrame) -> str:
    out = []
    for metric in METRICS:
        sub = t[t["metric"] == metric]
        if sub.empty:
            continue
        hdr = (f"{'model':<34}{'bench':<7}{'tasks':>6}{'items':>7}"
               f"{'step1':>10}{'step2':>10}{'improve%':>10}{'win%':>7}{'ident':>7}")
        out += ["", f"── {metric} " + "─" * max(0, len(hdr) - len(metric) - 4),
                hdr, "-" * len(hdr)]
        for _, r in sub.sort_values(["model", "benchmark"]).iterrows():
            out.append(
                f"{str(r['model'])[-34:]:<34}{r['benchmark']:<7}"
                f"{r['n_tasks']:>6.0f}{r['n_items']:>7.0f}"
                f"{r[f'step1_{metric}']:>10.4f}{r[f'step2_{metric}']:>10.4f}"
                f"{r['improvement_pct']:>+10.2f}{r['win_rate'] * 100:>7.1f}"
                f"{r['n_identical']:>7.0f}")
    out += ["",
            "improve% is (step1 - step2)/step1 per task, weighted by scored",
            "items, so POSITIVE means the pseudo-known-future pass helped.",
            "Both steps score the FIRST variate only, on the same items.",
            "ident counts items where the two steps returned the same forecast —",
            "a large count means the covariates did not reach the model, which",
            "reads as 'no gain' but is not one."]
    return "\n".join(out)


def plot(t: pd.DataFrame, out_png: Path) -> None:
    """Improvement per model, grouped by benchmark, one panel per metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in METRICS if not t[t["metric"] == m].empty]
    if not metrics:
        return
    models = sorted(t["model"].unique())
    benches = sorted(t["benchmark"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.4 * len(metrics), 4.4),
                             squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        sub = t[t["metric"] == metric]
        x = np.arange(len(models))
        w = 0.8 / max(len(benches), 1)
        for i, b in enumerate(benches):
            vals = [sub[(sub["model"] == m) & (sub["benchmark"] == b)]
                    ["improvement_pct"].mean() for m in models]
            ax.bar(x + i * w - 0.4 + w / 2, vals, width=w, label=b)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(m).split("/")[-1][:18] for m in models],
                           fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(f"{metric} improvement, step1→step2 (%)", fontsize=10)
        ax.set_title(f"{metric}: does a pseudo known-future covariate help?",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("positive = feeding the model its own covariate forecast "
                 "helped; first variate scored only", fontsize=8, y=0.01)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    table = build(load(args.results))
    if table.empty:
        raise SystemExit("no scorable rows in the per-model csvs")
    dest = args.results / "exp005_table.csv"
    table.to_csv(dest, index=False)
    png = args.results / "exp005_improvement.png"
    plot(table, png)
    print(render(table))
    logger.info("\nwrote %s\nwrote %s", dest, png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
