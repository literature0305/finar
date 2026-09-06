#!/usr/bin/env python3
"""FiNAR exp001 tables — iteration 1 vs iteration 2 over the (phi, rho, H) grid.

Reads the per-task csv every `run_eval.py` run leaves behind and reports, per
cell, what the second recursion pass bought over the first.

WHAT IS COMPARABLE AND WHAT IS NOT
The nine dependency cells do NOT share a difficulty — that is the deliberate
cost of making the dependency identifiable (see build_dataset.py, "WHAT IS AND
IS NOT HELD FIXED"). So raw MASE may be read DOWN a column, never ACROSS one.
The comparable statistic is excess risk over the cell's own analytic oracle,

    excess_k = MSE_k - oracle,     oracle = sigma^2 (1 - phi)

and the quantity the hypothesis is about is how much of that excess the second
pass removes:

    closed = (excess_1 - excess_2) / excess_1

`closed` is a share of the removable error, so it IS comparable across cells,
and it is what the FiNAR claim predicts should grow with phi, with rho and
with H. Raw MASE, its gap, and the win rate are reported beside it because
they are what the concept note's existing tables use.

WIN RATE is per task, iteration 2 against iteration 1, ties counting half —
the same definition as the repo's `iteration_performance.py`, so numbers from
here and from there mean the same thing.

Usage
-----
    python build_table.py results/                      # every model found
    python build_table.py results/eo-v4-K2 --iters 1 2
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
#: A cell, optionally with the `_sNN` shard suffix build_dataset.py appends.
#: Shards are the win rate's denominator — the evaluator emits one row per
#: dataset, so without them a cell is a single row and a win rate over it is
#: 0 or 1 rather than a rate.
_CELL_RE = re.compile(
    r"^H(\d+)_phi(low|mid|high)_rho(low|mid|high)(?:_s(\d+))?$")

LEVELS = ("low", "mid", "high")


def find_runs(root: Path) -> dict[str, Path]:
    """``{model name: per-task csv}`` for every run under ``root``."""
    out = {}
    for csv in sorted(root.rglob("finar_exp001.csv")):
        out[csv.parent.name] = csv
    return out


def depth_columns(df: pd.DataFrame) -> dict[int, dict[str, str]]:
    cols: dict[int, dict[str, str]] = {}
    for c in df.columns:
        m = _REPEAT_RE.match(str(c))
        if m:
            cols.setdefault(int(m.group(1)), {})[m.group(2)] = c
    return cols


def parse_cell(name: str) -> tuple[int, str, str] | None:
    m = _CELL_RE.match(str(name))
    return (int(m.group(1)), m.group(2), m.group(3)) if m else None


def base_cell(name: str) -> str | None:
    """``H16_philow_rholow_s03`` -> ``H16_philow_rholow``."""
    m = _CELL_RE.match(str(name))
    return None if m is None else f"H{m.group(1)}_phi{m.group(2)}_rho{m.group(3)}"


def win_rate(lo: np.ndarray, hi: np.ndarray) -> tuple[float, int]:
    """Share of tasks where the deeper pass beats the shallower; ties half.

    Pairwise complete cases — a task counts only when BOTH depths scored it, so
    one depth failing a handful of tasks does not shift the other's mean.
    """
    both = np.isfinite(lo) & np.isfinite(hi)
    n = int(both.sum())
    if n == 0:
        return float("nan"), 0
    a, b = lo[both], hi[both]
    return float(((b < a) + 0.5 * (b == a)).mean()), n


def build(df: pd.DataFrame, diag: dict, lo: int, hi: int) -> pd.DataFrame:
    """One row per cell.

    Two shapes, chosen by what the csv carries. With per-depth columns it is
    the iteration comparison — both depths, their gap, the win rate. Without
    them it is a BASELINE table of the headline metric over the same grid,
    which is what a model with no iteration axis (Chronos-2) can answer, and
    what makes its numbers comparable cell-for-cell with the EO run's
    iteration 1.
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
    df["_cell"] = df[ds_col].map(base_cell)
    unknown = df[df["_cell"].isna()][ds_col].unique()
    if len(unknown):
        logger.warning("skipping %d unrecognised dataset(s): %s",
                       len(unknown), list(unknown)[:3])

    rows = []
    for cell, grp in df.dropna(subset=["_cell"]).groupby("_cell"):
        parsed = parse_cell(cell)
        if parsed is None:
            continue
        H, phi, rho = parsed
        d = diag.get("cells", {}).get(str(cell), {})
        oracle = d.get("oracle_mse")

        row = {"cell": cell, "H": H, "phi": phi, "rho": rho,
               "oracle_mse": oracle,
               "explained": d.get("measured_explained_frac"),
               "xvar_corr": d.get("measured_cross_variate_corr")}
        if baseline:
            for metric in ("MASE", "WQL"):
                if metric not in grp.columns:
                    continue
                v = pd.to_numeric(grp[metric], errors="coerce").to_numpy(float)
                ok = np.isfinite(v)
                if ok.any():
                    row[metric] = float(v[ok].mean())
                    # Spread ACROSS SHARDS, not across series: it is the
                    # sampling error of this cell's mean, which is what says
                    # whether a difference between cells is real.
                    row[f"{metric}_sd"] = float(v[ok].std(ddof=1)) if ok.sum() > 1 else np.nan
                    row[f"n_{metric}"] = int(ok.sum())
            rows.append(row)
            continue
        for metric in ("MASE", "WQL"):
            if metric not in cols[lo] or metric not in cols[hi]:
                continue
            a = pd.to_numeric(grp[cols[lo][metric]], errors="coerce").to_numpy(float)
            b = pd.to_numeric(grp[cols[hi][metric]], errors="coerce").to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            if not both.any():
                continue
            am, bm = float(a[both].mean()), float(b[both].mean())
            row[f"iter{lo}_{metric}"] = am
            row[f"iter{hi}_{metric}"] = bm
            row[f"gap_{metric}"] = am - bm
            row[f"gap_pct_{metric}"] = (am - bm) / am * 100 if am else np.nan
            w, n = win_rate(a, b)
            row[f"win_{metric}"], row[f"n_{metric}"] = w, n

        # Excess risk needs an MSE, and the csv carries MASE/WQL. MASE is a
        # scaled ABSOLUTE error, so it is not squared error and cannot be turned
        # into one; the excess column is therefore computed only when the run
        # also emitted a squared-error column. Left blank rather than faked.
        for depth, tag in ((lo, "1"), (hi, "2")):
            mse_col = f"repeat{depth}_MSE"
            if mse_col in grp.columns and oracle:
                m = pd.to_numeric(grp[mse_col], errors="coerce").mean()
                row[f"excess_{tag}"] = float(m) - oracle
        if "excess_1" in row and "excess_2" in row and row["excess_1"]:
            row["closed_frac"] = (row["excess_1"] - row["excess_2"]) / row["excess_1"]
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    order = {lv: i for i, lv in enumerate(LEVELS)}
    return (pd.DataFrame(rows)
            .sort_values(["H", "phi", "rho"],
                         key=lambda s: s.map(order) if s.name in ("phi", "rho") else s)
            .reset_index(drop=True))


def render_baseline(t: pd.DataFrame, metric: str) -> str:
    """The grid for a model with no iteration axis: one value per cell."""
    if metric not in t.columns:
        return ""
    out = [f"\n{'=' * 60}", f"{metric} (no iteration axis)", "=" * 60]
    for H in sorted(t["H"].unique()):
        sub = t[t["H"] == H]
        out.append(f"\nH = {H}")
        out.append(f"  {'phi\\rho':<10} " + "".join(f"{r:>14}" for r in LEVELS))
        for phi in LEVELS:
            cells = []
            for rho in LEVELS:
                r = sub[(sub["phi"] == phi) & (sub["rho"] == rho)]
                if r.empty or pd.isna(r.iloc[0].get(metric)):
                    cells.append(f"{'-':>14}")
                    continue
                r = r.iloc[0]
                cells.append(f"{r[metric]:>8.4f}"
                             f"±{r.get(f'{metric}_sd', float('nan')):<5.3f}")
            out.append(f"  {phi:<10} " + "".join(cells))
    out += ["", "mean over shards ± sd across shards. Cells have different",
            "oracles, so compare a cell to the oracle column in the csv, not",
            "to its neighbours."]
    return "\n".join(out)


def render(t: pd.DataFrame, lo: int, hi: int, metric: str) -> str:
    """The grid as text: one block per horizon, phi down, rho across."""
    a, b = f"iter{lo}_{metric}", f"iter{hi}_{metric}"
    if a not in t.columns:
        return ""
    out = [f"\n{'=' * 78}", f"{metric}: iteration {lo} -> {hi}", "=" * 78]
    for H in sorted(t["H"].unique()):
        sub = t[t["H"] == H]
        out.append(f"\nH = {H}")
        out.append(f"  {'phi\\rho':<10} " +
                   "".join(f"{r:>22}" for r in LEVELS))
        for phi in LEVELS:
            cells = []
            for rho in LEVELS:
                r = sub[(sub["phi"] == phi) & (sub["rho"] == rho)]
                if r.empty or pd.isna(r.iloc[0][a]):
                    cells.append(f"{'-':>22}")
                    continue
                r = r.iloc[0]
                cells.append(f"{r[a]:>7.4f}->{r[b]:<7.4f}"
                             f"{r.get(f'win_{metric}', float('nan')) * 100:>6.0f}%")
            out.append(f"  {phi:<10} " + "".join(cells))
    out += [
        "",
        "cell = iter1 MASE -> iter2 MASE, then iteration-2 win rate.",
        "READ DOWN AND ACROSS WITHIN A HORIZON ONLY FOR THE WIN RATE AND THE",
        "GAP: the nine cells have different oracles, so raw levels are not",
        "comparable between cells. See closed_frac in the csv for the",
        "cross-cell statistic.",
    ]
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
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    diag_path = args.data_root / "diagnostics.json"
    diag = json.loads(diag_path.read_text()) if diag_path.is_file() else {}
    if not diag:
        logger.warning("no diagnostics.json at %s — oracle and excess-risk "
                       "columns will be blank", diag_path)

    runs = find_runs(args.results)
    if not runs:
        raise SystemExit(f"no finar_exp001.csv under {args.results}")

    lo, hi = args.iters
    ok = False
    for model, csv in runs.items():
        logger.info("\n### %s  (%s)", model, csv)
        table = build(pd.read_csv(csv), diag, lo, hi)
        if table.empty:
            logger.warning("  no recognised cells")
            continue
        dest = csv.parent / "finar_exp001_table.csv"
        table.to_csv(dest, index=False)
        for metric in ("MASE", "WQL"):
            print(render(table, lo, hi, metric)
                  or render_baseline(table, metric))
        logger.info("wrote %s", dest)
        ok = True
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
