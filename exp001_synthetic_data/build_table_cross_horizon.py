#!/usr/bin/env python3
"""Long-format table and heatmaps for the equal-observed cross-horizon sweep.

Reads `results.csv` (one row per task, `repeat{r}_MASE` / `repeat{r}_WQL`
columns) and writes:

    cross_horizon_table.csv        horizon x regime x iteration, long format
    cross_horizon_observed.csv     item 9: horizon vs observed length, both corpora
    cross_horizon_heatmaps.png     regime x iteration, and horizon x iteration

The two heatmaps are the two 2-D faces of a 3-D result. Averaging is by the
axis NOT shown: the regime x iteration panel averages over the four horizons,
the horizon x iteration panel over the three regimes. Each is a mean of MASE
values that already share a denominator — the source corpus computes the
seasonal-naive scale from the observed context, which is identical across all
twelve tasks — so the cells are commensurable.

Usage
-----
    python build_table_cross_horizon.py <results-dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("finar_exp001_2_table")

REGIME_ORDER = ["high", "mid", "low"]
_NAME_RE = re.compile(r"h(?P<h>\d+)/(?P<regime>high|mid|low)\s*$")
_REPEAT_RE = re.compile(r"^repeat(?P<r>\d+)_(?P<metric>WQL|MASE)$")


def melt(df: pd.DataFrame, trained_depth: int) -> pd.DataFrame:
    """One row per (horizon, regime, iteration)."""
    name_col = next((c for c in ("dataset", "name", "task", "dataset_name")
                     if c in df.columns), df.columns[0])
    depths = sorted({int(m["r"]) for c in df.columns
                     if (m := _REPEAT_RE.match(c))})
    if not depths:
        raise SystemExit(
            "no repeat<r>_MASE columns in results.csv — the run scored a "
            "single depth, so there are no per-iteration numbers to table")
    rows = []
    for _, r in df.iterrows():
        m = _NAME_RE.search(str(r[name_col]))
        if not m:
            logger.warning("skipping unrecognised task name %r", r[name_col])
            continue
        for d in depths:
            rows.append({
                "horizon": int(m["h"]), "regime": m["regime"], "iteration": d,
                "MASE": float(r.get(f"repeat{d}_MASE", np.nan)),
                "WQL": float(r.get(f"repeat{d}_WQL", np.nan)),
                # Depths past the trained maximum are still recorded — that is
                # the question — but a reader must not mistake them for
                # in-distribution numbers.
                "beyond_trained_depth": d > trained_depth,
            })
    out = pd.DataFrame(rows)
    out["regime"] = pd.Categorical(out["regime"], REGIME_ORDER, ordered=True)
    return out.sort_values(["horizon", "regime", "iteration"])


def heatmaps(t: pd.DataFrame, dest: Path, metric: str, trained_depth: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    faces = [
        ("regime", REGIME_ORDER, "dependency regime", "averaged over 4 horizons"),
        ("horizon", sorted(t["horizon"].unique()), "horizon",
         "averaged over 3 regimes"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, (axis, order, label, note) in zip(axes, faces):
        piv = (t.pivot_table(index=axis, columns="iteration", values=metric,
                             aggfunc="mean", observed=False)
                .reindex(order))
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel("iteration (recursion depth)")
        ax.set_ylabel(label)
        ax.set_title(f"{metric} by {label} x iteration\n({note})", fontsize=10)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=8, color="w")
        # The trained/extrapolated boundary, drawn once rather than explained
        # in a caption nobody reads next to the numbers.
        if trained_depth < max(piv.columns):
            ax.axvline(list(piv.columns).index(trained_depth) + 0.5,
                       color="r", lw=1.5, ls="--")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"equal observed history (1040 steps) at every horizon; "
                 f"red line = trained depth {trained_depth}, right of it is "
                 f"deeper than training", fontsize=9, y=0.02)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render(t: pd.DataFrame, metric: str) -> str:
    piv = t.pivot_table(index=["horizon", "regime"], columns="iteration",
                        values=metric, aggfunc="mean", observed=False)
    out = [f"── {metric} ─────────────────────────────────────────────",
           "horizon regime  " + "".join(f"{f'iter{c}':>10}" for c in piv.columns)
           + f"{'iter1→N':>10}"]
    for (h, regime), row in piv.iterrows():
        gain = ((row.iloc[0] - row.iloc[-1]) / row.iloc[0] * 100
                if np.isfinite(row.iloc[0]) and row.iloc[0] else np.nan)
        out.append(f"{h:>7} {regime:<7}"
                   + "".join(f"{v:>10.4f}" for v in row.values)
                   + f"{gain:>+9.2f}%")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    man = json.loads((args.results / "manifest.json").read_text())
    trained = int(man.get("trained_depth", 1))
    df = pd.read_csv(args.results / "results.csv")
    t = melt(df, trained)
    t.to_csv(args.results / "cross_horizon_table.csv", index=False)
    pd.DataFrame(man["observed_report"]).to_csv(
        args.results / "cross_horizon_observed.csv", index=False)
    heatmaps(t, args.results / "cross_horizon_heatmaps.png", "MASE", trained)

    print(render(t, "MASE"))
    print()
    print(render(t, "WQL"))
    print("\n── observed history the model actually receives ──")
    print(pd.DataFrame(man["observed_report"]).to_string(index=False))
    n_ex = int(t["beyond_trained_depth"].sum())
    if n_ex:
        print(f"\n{n_ex} of {len(t)} rows are DEEPER than the checkpoint's "
              f"trained depth ({trained}); flagged in the csv.")
    logger.info("\nwrote %s\nwrote %s\nwrote %s",
                args.results / "cross_horizon_table.csv",
                args.results / "cross_horizon_observed.csv",
                args.results / "cross_horizon_heatmaps.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
