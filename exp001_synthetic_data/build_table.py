#!/usr/bin/env python3
"""FiNAR exp001 tables — iteration 1 vs iteration 2 over (phi, shuffle, group, H).

Reads the per-task csv every `run_eval.py` run leaves behind and reports, per
cell, what the second recursion pass bought over the first.

TWO TABLES, ONE GRID
The corpus crosses three factors (`build_dataset.py`): dependency along time
(`phi`), and two different ways of removing cross-variate dependency —
re-pairing variates across items (`shuffle`) and shrinking the group that
shares a latent bank (`group`). Both tables are marginals of the same 27-cell
grid, each holding the other cross-variate axis at the level where the
dependency is intact:

    shuffle table    rows phi, columns none / half / all      at group = full
    group table      rows phi, columns full / half / no       at shuffle = none

They must agree at their zero ends — `all` shuffle and `no` group reach a
cross-variate correlation of 0.00 by different routes — and that agreement is
the design's own consistency check, not a redundancy.

WHAT THE CELL SHOWS
`iteration-1 metric -> iteration-2 metric`, then the IMPROVEMENT the second
pass bought:

    improvement = (metric_1 - metric_2) / metric_1

computed PER SHARD and then averaged, so the +- beside it is the spread over
shards and says whether an improvement is distinguishable from noise. That
matters more here than it looks: on the previous build the shard spread was
0.3-2% of the metric, which is the same order as the improvements being
reported. The win rate is still written to the csv for the same reason — it is
the paired statistic, and it survives a cell whose mean moves by less than its
spread.

COMPARING CELLS
Improvement is a ratio, so it is not tied to a cell's difficulty the way a raw
metric is, and it may be read across the grid. The raw levels may not: cells
differ in how much of the series is predictable at all (`phi`), and this build
has no analytic oracle to subtract — the previous one did, and it stopped being
valid the moment the context was truncated to a fixed window, because the
achievable floor then depends on the period mix rather than on `phi`. Read the
arrows within a cell and the percentages across cells.

Usage
-----
    python build_table.py results/                      # every model found
    python build_table.py results/eo-v4-K2 --iters 1 2
    python build_table.py results/ --at-group half --at-shuffle half
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("finar_exp001_table")

_REPEAT_RE = re.compile(r"^repeat(\d+)_(MASE|WQL)$")

#: A dataset name as `run_eval.py` writes it: the cell, the shard that gives a
#: dispersion statistic its denominator, and the view suffix that carries the
#: evaluation protocol (context and horizon).
_CELL_RE = re.compile(
    r"^phi(low|mid|high)_sh(none|half|all)_gp(full|half|no)"
    r"(?:_s(\d+))?(?:_c(\d+)h(\d+))?$")

PHI_LEVELS = ("low", "mid", "high")
SHUFFLE_LEVELS = ("none", "half", "all")
GROUP_LEVELS = ("full", "half", "no")

#: Which axis each table varies, and which one it holds fixed.
AXES = {"shuffle": ("shuf", SHUFFLE_LEVELS, "group"),
        "group": ("group", GROUP_LEVELS, "shuf")}


def find_runs(root: Path) -> dict[str, Path]:
    """``{model name: per-task csv}`` for every run under ``root``."""
    return {csv.parent.name: csv
            for csv in sorted(root.rglob("finar_exp001.csv"))}


def depth_columns(df: pd.DataFrame) -> dict[int, dict[str, str]]:
    cols: dict[int, dict[str, str]] = {}
    for c in df.columns:
        m = _REPEAT_RE.match(str(c))
        if m:
            cols.setdefault(int(m.group(1)), {})[m.group(2)] = c
    return cols


def parse_cell(name: str):
    """``phihigh_shall_gpfull_s00_c512h256`` -> parts, or None."""
    m = _CELL_RE.match(str(name))
    if m is None:
        return None
    return {"phi": m.group(1), "shuf": m.group(2), "group": m.group(3),
            "shard": m.group(4),
            "context": int(m.group(5)) if m.group(5) else None,
            "H": int(m.group(6)) if m.group(6) else None,
            "diag_key": f"phi{m.group(1)}_sh{m.group(2)}_gp{m.group(3)}"}


def win_rate(lo: np.ndarray, hi: np.ndarray) -> tuple[float, int]:
    """Share of tasks where the deeper pass beats the shallower; ties half.

    Pairwise complete cases — a task counts only when BOTH depths scored it, so
    one depth failing a handful of tasks does not shift the other's mean. Same
    definition as the repo's `iteration_performance.py`.
    """
    both = np.isfinite(lo) & np.isfinite(hi)
    n = int(both.sum())
    if n == 0:
        return float("nan"), 0
    a, b = lo[both], hi[both]
    return float(((b < a) + 0.5 * (b == a)).mean()), n


def build(df: pd.DataFrame, diag: dict, lo: int, hi: int) -> pd.DataFrame:
    """One row per (cell, horizon).

    Two shapes, chosen by what the csv carries. With per-depth columns it is
    the iteration comparison. Without them it is a BASELINE table of the
    headline metric over the same grid, which is what a model with no iteration
    axis (Chronos-2) can answer, and what makes its numbers comparable
    cell-for-cell with the EO run's iteration 1.
    """
    cols = depth_columns(df)
    baseline = lo not in cols or hi not in cols
    if baseline and cols:
        raise SystemExit(
            f"the csv has depths {sorted(cols)} but not both {lo} and {hi}. "
            f"A run with a single depth means the checkpoint reported "
            f"report_depth()==1 — see run_exp001.sh, DEPTH.")
    if baseline:
        logger.info("  no per-depth columns — baseline table (no iteration axis)")

    ds_col = next((c for c in ("dataset", "name", "dataset_config")
                   if c in df.columns), None)
    if ds_col is None:
        raise SystemExit(f"no dataset column in csv (have {list(df.columns)})")

    df = df.copy()
    parts = df[ds_col].map(parse_cell)
    unknown = df[parts.isna()][ds_col].unique()
    if len(unknown):
        logger.warning("skipping %d unrecognised dataset(s): %s",
                       len(unknown), list(unknown)[:3])
    df = df[parts.notna()].copy()
    parts = parts[parts.notna()]
    for k in ("phi", "shuf", "group", "context", "H", "diag_key"):
        df["_" + k] = [p[k] for p in parts]

    rows = []
    keys = ["_phi", "_shuf", "_group", "_H", "_context", "_diag_key"]
    for (phi, shuf, group, H, ctx, dkey), grp in df.groupby(keys, dropna=False):
        d = diag.get("cells", {}).get(str(dkey), {})
        row = {"cell": dkey, "H": H, "context": ctx,
               "phi": phi, "shuf": shuf, "group": group,
               "observed_corr": d.get("observed_corr_signed"),
               "observed_corr_abs": d.get("observed_corr_abs"),
               "explained": d.get("measured_explained_frac"),
               "cross_channel_gain": d.get("cross_channel_gain")}
        for metric in ("MASE", "WQL"):
            if baseline:
                if metric not in grp.columns:
                    continue
                v = pd.to_numeric(grp[metric], errors="coerce").to_numpy(float)
                ok = np.isfinite(v)
                if ok.any():
                    row[metric] = float(v[ok].mean())
                    # Spread ACROSS SHARDS, not across series: it is the
                    # sampling error of this cell's mean, which is what says
                    # whether a difference between cells is real.
                    row[f"{metric}_sd"] = (float(v[ok].std(ddof=1))
                                           if ok.sum() > 1 else np.nan)
                    row[f"n_{metric}"] = int(ok.sum())
                continue
            if metric not in cols[lo] or metric not in cols[hi]:
                continue
            a = pd.to_numeric(grp[cols[lo][metric]],
                              errors="coerce").to_numpy(float)
            b = pd.to_numeric(grp[cols[hi][metric]],
                              errors="coerce").to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b) & (a != 0)
            if not both.any():
                continue
            am, bm = float(a[both].mean()), float(b[both].mean())
            # PER-SHARD improvement, then averaged. The ratio of the means and
            # the mean of the ratios are different numbers, and only the second
            # has a spread that can be quoted beside it.
            imp = (a[both] - b[both]) / a[both] * 100.0
            row[f"iter{lo}_{metric}"] = am
            row[f"iter{hi}_{metric}"] = bm
            row[f"imp_pct_{metric}"] = float(imp.mean())
            row[f"imp_pct_sd_{metric}"] = (float(imp.std(ddof=1))
                                           if imp.size > 1 else np.nan)
            w, n = win_rate(a, b)
            row[f"win_{metric}"], row[f"n_{metric}"] = w, n
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    o_phi = {lv: i for i, lv in enumerate(PHI_LEVELS)}
    o_sh = {lv: i for i, lv in enumerate(SHUFFLE_LEVELS)}
    o_gp = {lv: i for i, lv in enumerate(GROUP_LEVELS)}
    order = {"phi": o_phi, "shuf": o_sh, "group": o_gp}
    return (pd.DataFrame(rows)
            .sort_values(["H", "phi", "shuf", "group"],
                         key=lambda s: s.map(order[s.name])
                         if s.name in order else s)
            .reset_index(drop=True))


def _corr_legend(t: pd.DataFrame, axis: str, levels, fixed: str,
                 at: str) -> str:
    """The observed cross-variate correlation each column actually carries.

    Printed with the table because the column headings are level names, and a
    level name is a claim about the data that the previous build got wrong by a
    factor of ten. These numbers are measured on the saved series.
    """
    sub = t[t[fixed] == at]
    out = []
    for lv in levels:
        # A RANGE over the phi rows, not one row's value: the correlation this
        # column carries depends on phi as well (the factor is scaled by
        # sqrt(phi)), so quoting a single number would misdescribe two of the
        # three rows it sits above.
        v = sub[sub[axis] == lv]["observed_corr"].dropna().unique()
        out.append(f"{lv}={min(v):+.2f}..{max(v):+.2f}" if len(v)
                   else f"{lv}=?")
    return "observed cross-variate corr (over the phi rows): " + "  ".join(out)


def render(t: pd.DataFrame, table: str, at: str, lo: int, hi: int,
           metric: str) -> str:
    """One block per horizon: phi down the rows, the chosen axis across."""
    axis, levels, fixed = AXES[table]
    a, b = f"iter{lo}_{metric}", f"iter{hi}_{metric}"
    if a not in t.columns:
        return ""
    sub = t[t[fixed] == at]
    if sub.empty:
        return ""
    head = f"{metric}: iteration {lo} -> {hi}   [{table} axis, {fixed}={at}]"
    out = [f"\n{'=' * 82}", head, "=" * 82]
    for H in sorted(sub["H"].dropna().unique()):
        blk = sub[sub["H"] == H]
        out.append(f"\nH = {int(H)}")
        out.append(f"  {'phi\\' + table:<12} "
                   + "".join(f"{lv:>22}" for lv in levels))
        for phi in PHI_LEVELS:
            cells = []
            for lv in levels:
                r = blk[(blk["phi"] == phi) & (blk[axis] == lv)]
                if r.empty or pd.isna(r.iloc[0][a]):
                    cells.append(f"{'-':>22}")
                    continue
                r = r.iloc[0]
                cells.append(f"{r[a]:>7.4f}->{r[b]:<7.4f}"
                             f"{r.get(f'imp_pct_{metric}', np.nan):>+6.1f}%")
            out.append(f"  {phi:<12} " + "".join(cells))
    out += ["", "cell = iter1 -> iter2, then the iteration-2 improvement "
                "(mean over shards).",
            _corr_legend(sub, axis, levels, fixed, at),
            "Improvement is a ratio and may be read across cells; the raw "
            "levels may not —",
            "cells differ in how much of the series is predictable at all. "
            "Per-shard spread",
            "and win rate are in the csv."]
    return "\n".join(out)


def render_baseline(t: pd.DataFrame, table: str, at: str, metric: str) -> str:
    """The grid for a model with no iteration axis: one value per cell."""
    axis, levels, fixed = AXES[table]
    if metric not in t.columns:
        return ""
    sub = t[t[fixed] == at]
    if sub.empty:
        return ""
    out = [f"\n{'=' * 64}",
           f"{metric} (no iteration axis)   [{table} axis, {fixed}={at}]",
           "=" * 64]
    for H in sorted(sub["H"].dropna().unique()):
        blk = sub[sub["H"] == H]
        out.append(f"\nH = {int(H)}")
        out.append(f"  {'phi\\' + table:<12} "
                   + "".join(f"{lv:>15}" for lv in levels))
        for phi in PHI_LEVELS:
            cells = []
            for lv in levels:
                r = blk[(blk["phi"] == phi) & (blk[axis] == lv)]
                if r.empty or pd.isna(r.iloc[0].get(metric)):
                    cells.append(f"{'-':>15}")
                    continue
                r = r.iloc[0]
                cells.append(f"{r[metric]:>9.4f}"
                             f"±{r.get(f'{metric}_sd', np.nan):<5.3f}")
            out.append(f"  {phi:<12} " + "".join(cells))
    out += ["", _corr_legend(sub, axis, levels, fixed, at),
            "mean over shards ± sd across shards."]
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path)
    p.add_argument("--data-root", type=Path,
                   default=Path("/group-volume/ts-dataset/finar_exp001"))
    p.add_argument("--iters", nargs=2, type=int, default=[1, 2],
                   metavar=("LO", "HI"))
    p.add_argument("--at-group", default="full", choices=GROUP_LEVELS,
                   help="group level the SHUFFLE table holds fixed")
    p.add_argument("--at-shuffle", default="none", choices=SHUFFLE_LEVELS,
                   help="shuffle level the GROUP table holds fixed")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    diag_path = args.data_root / "diagnostics.json"
    diag = json.loads(diag_path.read_text()) if diag_path.is_file() else {}
    if not diag:
        logger.warning("no diagnostics.json at %s — the correlation legend "
                       "will be blank", diag_path)

    runs = find_runs(args.results)
    if not runs:
        raise SystemExit(f"no finar_exp001.csv under {args.results}")

    lo, hi = args.iters
    fixed_at = {"shuffle": args.at_group, "group": args.at_shuffle}
    ok = False
    for model, csv in runs.items():
        logger.info("\n### %s  (%s)", model, csv)
        table = build(pd.read_csv(csv), diag, lo, hi)
        if table.empty:
            logger.warning("  no recognised cells")
            continue
        dest = csv.parent / "finar_exp001_table.csv"
        table.to_csv(dest, index=False)
        for which in ("shuffle", "group"):
            at = fixed_at[which]
            for metric in ("MASE", "WQL"):
                print(render(table, which, at, lo, hi, metric)
                      or render_baseline(table, which, at, metric))
        logger.info("wrote %s", dest)
        ok = True
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
