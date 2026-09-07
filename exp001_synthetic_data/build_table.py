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

#: The level names, IMPORTED from the module that generates the corpora. A
#: level added there but not here would be scored, written to the csv, and then
#: dropped from every printed table without a warning.
from build_dataset import (  # noqa: E402  (a sibling module, same directory)
    GROUP_SIZES, PHI_LEVELS as _PHI, SHUFFLE_LEVELS as _SHUF, cell_name,
)

PHI_LEVELS = tuple(_PHI)
SHUFFLE_LEVELS = tuple(_SHUF)
GROUP_LEVELS = tuple(GROUP_SIZES)

#: A dataset name as `run_eval.py` writes it: the cell, the shard that gives a
#: dispersion statistic its denominator, and the view suffix that carries the
#: evaluation protocol (context and horizon). BUILT from the level tuples above
#: so the grammar cannot fall out of step with them — a stale alternation would
#: not raise, it would silently drop rows into the "unrecognised" warning.
_CELL_RE = re.compile(
    rf"^phi({'|'.join(PHI_LEVELS)})_sh({'|'.join(SHUFFLE_LEVELS)})"
    rf"_gp({'|'.join(GROUP_LEVELS)})(?:_s(\d+))?(?:_c(\d+)h(\d+))?$")

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
            "context": int(m.group(5)) if m.group(5) else None,
            "H": int(m.group(6)) if m.group(6) else None}


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
    for k in ("phi", "shuf", "group", "context", "H"):
        df["_" + k] = [p[k] for p in parts]

    rows = []
    keys = ["_phi", "_shuf", "_group", "_H", "_context"]
    for (phi, shuf, group, H, ctx), grp in df.groupby(keys, dropna=False):
        # `cell_name` rather than re-concatenating the regex groups: the
        # generator owns this spelling and diagnostics.json is keyed by it.
        dkey = cell_name(phi, shuf, group)
        d = diag.get("cells", {}).get(dkey, {})
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
    order = {col: {lv: i for i, lv in enumerate(levels)}
             for col, levels in (("phi", PHI_LEVELS), ("shuf", SHUFFLE_LEVELS),
                                 ("group", GROUP_LEVELS))}
    return (pd.DataFrame(rows)
            .sort_values(["H", "phi", "shuf", "group"],
                         key=lambda s: s.map(order[s.name])
                         if s.name in order else s)
            .reset_index(drop=True))


def _corr_legend(sub: pd.DataFrame, axis: str, levels) -> str:
    """The observed cross-variate correlation each column actually carries.

    Printed with the table because the column headings are level names, and a
    level name is a claim about the data. A RANGE over the phi rows, not one
    row's value: the correlation a column carries depends on phi as well (the
    factor is scaled by sqrt(phi)), so a single number would misdescribe two of
    the three rows it sits above.
    """
    out = []
    for lv in levels:
        v = sub[sub[axis] == lv]["observed_corr"].dropna().unique()
        out.append(f"{lv}={min(v):+.2f}..{max(v):+.2f}" if len(v)
                   else f"{lv}=?")
    return "observed cross-variate corr (over the phi rows): " + "  ".join(out)


def _render_grid(t: pd.DataFrame, table: str, at: str, guard: str, title: str,
                 rule: int, cell_w: int, fmt, footer: list[str]) -> str:
    """One block per horizon: phi down the rows, the chosen axis across.

    THE grid renderer. The iteration table and the baseline table differ only
    in which column has to be present, how wide a cell is, and what goes in it;
    everything else — the axis lookup, the per-horizon blocks, the header row,
    the missing-cell dash, the correlation legend — was written twice and had
    already started to drift apart within one commit.
    """
    axis, levels, fixed = AXES[table]
    if guard not in t.columns:
        return ""
    sub = t[t[fixed] == at]
    if sub.empty:
        return ""
    out = [f"\n{'=' * rule}", f"{title}   [{table} axis, {fixed}={at}]",
           "=" * rule]
    for H in sorted(sub["H"].dropna().unique()):
        blk = sub[sub["H"] == H]
        out.append(f"\nH = {int(H)}")
        out.append(f"  {'phi\\' + table:<12} "
                   + "".join(f"{lv:>{cell_w}}" for lv in levels))
        for phi in PHI_LEVELS:
            cells = []
            for lv in levels:
                r = blk[(blk["phi"] == phi) & (blk[axis] == lv)]
                if r.empty or pd.isna(r.iloc[0][guard]):
                    cells.append(f"{'-':>{cell_w}}")
                else:
                    cells.append(fmt(r.iloc[0]))
            out.append(f"  {phi:<12} " + "".join(cells))
    return "\n".join(out + [""] + footer + [_corr_legend(sub, axis, levels)])


def render(t: pd.DataFrame, table: str, at: str, lo: int, hi: int,
           metric: str) -> str:
    """The iteration comparison: iter1 -> iter2 and what the second pass bought."""
    a, b = f"iter{lo}_{metric}", f"iter{hi}_{metric}"
    return _render_grid(
        t, table, at, guard=a, rule=82, cell_w=22,
        title=f"{metric}: iteration {lo} -> {hi}",
        fmt=lambda r: (f"{r[a]:>7.4f}->{r[b]:<7.4f}"
                       f"{r.get(f'imp_pct_{metric}', np.nan):>+6.1f}%"),
        footer=["cell = iter1 -> iter2, then the iteration-2 improvement "
                "(mean over shards).",
                "Improvement is a ratio and may be read across cells; the raw "
                "levels may not —",
                "cells differ in how much of the series is predictable at all. "
                "Per-shard spread",
                "and win rate are in the csv."])


def render_baseline(t: pd.DataFrame, table: str, at: str, metric: str) -> str:
    """The grid for a model with no iteration axis: one value per cell."""
    return _render_grid(
        t, table, at, guard=metric, rule=64, cell_w=15,
        title=f"{metric} (no iteration axis)",
        fmt=lambda r: (f"{r[metric]:>9.4f}"
                       f"±{r.get(f'{metric}_sd', np.nan):<5.3f}"),
        footer=["mean over shards ± sd across shards."])


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
