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
import sys
from pathlib import Path

# CPU CAP — AT MODULE SCOPE, ABOVE numpy/pandas/pyarrow. OpenMP and BLAS size
# their pools when the library first loads, so a cap applied later is ignored by
# them. Measured uncapped on an 18-core box: 17.9 effective cores. The launcher
# also exports these, but this file is runnable on its own. No
# --num-workers here: these parsers do not take one, and peeking for a
# flag argparse would then reject is a promise the script cannot keep.
sys.path.append(str(next(d for d in Path(__file__).resolve().parents
                         if (d / "finar_cpu.py").is_file())))
from finar_cpu import limit_cpu  # noqa: E402

limit_cpu(quiet=True)   # from $OMP_NUM_THREADS or the allocation

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_step import improvement_pct  # noqa: E402

logger = logging.getLogger("finar_exp005_table")

METRICS = ("MASE", "WQL")


def load(root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(c) for c in sorted(root.glob("*.csv"))
              if not c.name.endswith("_table.csv")]
    if not frames:
        raise SystemExit(f"no per-model csv under {root}")
    out = pd.concat(frames, ignore_index=True)
    # ADDITIVE, so the published pre-alpha results still table: a run from
    # before --alpha existed has no such column, and grouping on it KeyErrors
    # on exactly the csvs the README's verification table reports.
    if "alpha" not in out.columns:
        out["alpha"] = 0.0
    for c in ("n_unforecastable", "n_oracle_substituted", "n_same_as_prev_alpha"):
        if c not in out.columns:
            out[c] = 0
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, benchmark, metric), pooled over tasks.

    Pooled by ITEM COUNT, not by task: a 32-item GIFT task and a 10-item fev
    window carry different evidence and a plain task mean would weight them
    equally. `n_scored` is the weight the metrics were computed on.
    """
    rows = []
    for (model, bench, alpha), g in df.groupby(
            ["model", "benchmark", "alpha"], dropna=False):
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
            per_task = improvement_pct(a[ok], b[ok])
            rows.append({
                "model": model, "benchmark": bench, "alpha": float(alpha),
                "metric": metric,
                "n_tasks": int(ok.sum()), "n_items": int(w[ok].sum()),
                f"step1_{metric}": s1, f"step2_{metric}": s2,
                "improvement_pct": float(np.nansum(per_task * w[ok])
                                         / np.nansum(w[ok])),
                "win_rate": float(np.average(
                    g["win_rate"].to_numpy(float)[ok], weights=w[ok])),
                "n_identical": int(g["n_identical"].to_numpy()[ok].sum()),
                # Both can make two alpha columns describe different things, so
                # they travel with n_identical rather than living in a docstring.
                "n_unforecastable": int(g["n_unforecastable"].to_numpy()[ok].sum()),
                "n_oracle_substituted": int(
                    g["n_oracle_substituted"].to_numpy()[ok].sum()),
            })
    return pd.DataFrame(rows)


def render(t: pd.DataFrame) -> str:
    out = []
    for metric in METRICS:
        sub = t[t["metric"] == metric]
        if sub.empty:
            continue
        hdr = (f"{'model':<30}{'bench':<7}{'alpha':>6}{'tasks':>6}{'items':>7}"
               f"{'step1':>10}{'step2':>10}{'improve%':>10}{'win%':>7}{'ident':>7}")
        out += ["", f"── {metric} " + "─" * max(0, len(hdr) - len(metric) - 4),
                hdr, "-" * len(hdr)]
        for _, r in sub.sort_values(["model", "benchmark", "alpha"]).iterrows():
            out.append(
                f"{str(r['model'])[-30:]:<30}{r['benchmark']:<7}"
                f"{r['alpha']:>6.2f}{r['n_tasks']:>6.0f}{r['n_items']:>7.0f}"
                f"{r[f'step1_{metric}']:>10.4f}{r[f'step2_{metric}']:>10.4f}"
                f"{r['improvement_pct']:>+10.2f}{r['win_rate'] * 100:>7.1f}"
                f"{r['n_identical']:>7.0f}"
                + ("" if r["alpha"] == 0 else "   (oracle leakage)"))
    out += ["",
            "EVERY alpha > 0 ROW LEAKS THE COVARIATES' TRUE FUTURE — an upper",
            "bound on what a perfect covariate forecaster could buy, not a",
            "score. alpha = 0 is the only honest arm.",
            "improve% is (step1 - step2)/step1 per task, weighted by scored",
            "items, so POSITIVE means the pseudo-known-future pass helped.",
            "Both steps score the FIRST variate only, on the same items.",
            "ident counts items where the two steps returned the same forecast —",
            "a large count means the covariates did not reach the model, which",
            "reads as 'no gain' but is not one."]
    return "\n".join(out)


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out_png: Path) -> None:
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _model_label(m, n: int = 18) -> str:
    """One truncation, so two figures in one directory name a model the same."""
    return str(m).split("/")[-1][:n]


def plot(t: pd.DataFrame, out_png: Path) -> None:
    """Improvement per model, grouped by benchmark, one panel per metric.

    ALPHA = 0 ONLY. This is the figure the README documents as exp005's result,
    and alpha > 0 is label leakage on the covariates; averaging the two into one
    bar would silently report the mean of a control and an upper bound. The
    sweep has its own figure, `alpha_curves`.
    """
    t = t[t["alpha"] == 0.0]
    if t.empty:
        return
    plt = _pyplot()
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
        ax.set_xticklabels([_model_label(m) for m in models],
                           fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(f"{metric} improvement, step1→step2 (%)", fontsize=10)
        ax.set_title(f"{metric}: does a pseudo known-future covariate help?",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("positive = feeding the model its own covariate forecast "
                 "helped; first variate scored only", fontsize=8, y=0.01)
    _save(fig, out_png)


def alpha_curves(t: pd.DataFrame, out_png: Path) -> None:
    """MASE, improvement and win rate against alpha — the question's own axis.

    Lines rather than a heatmap: alpha is ordered and the reading is whether
    the curve is monotone, which a colour ramp hides. One panel per quantity,
    one line per (model, benchmark).
    """
    plt = _pyplot()
    sub = t[t["metric"] == "MASE"]
    if sub.empty or sub["alpha"].nunique() < 2:
        return
    panels = [("step2_MASE", "step-2 MASE (lower is better)", None, None),
              ("improvement_pct", "MASE improvement vs step 1 (%)", 0.0, "-"),
              ("win_rate", "win rate vs step 1", 0.5, ":")]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2))
    for ax, (col, title, ref, ls) in zip(axes, panels):
        for (model, bench), g in sub.groupby(["model", "benchmark"]):
            g = g.sort_values("alpha")
            ax.plot(g["alpha"], g[col], marker="o",
                    label=f"{_model_label(model)}/{bench}")
        if ref is not None:
            ax.axhline(ref, color="k", lw=0.8, ls=ls)
        ax.set_xlabel("alpha  (0 = pseudo covariate future, 1 = oracle)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("alpha blends the covariates' TRUE future into the model's own "
                 "forecast of them; the scored target is never oracle. "
                 "alpha > 0 is label leakage on the covariates — an upper "
                 "bound, not a score.", fontsize=8, y=0.005)
    _save(fig, out_png)


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
    apng = args.results / "exp005_alpha.png"
    alpha_curves(table, apng)
    print(render(table))
    logger.info("\nwrote %s\nwrote %s\nwrote %s", dest, png, apng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
