#!/usr/bin/env python3
"""What the second pass does with a better input — table and figure.

Reads the per-alpha csvs `run_eval.py` leaves under
``<results>/<benchmark>/alpha<a>/results.csv`` and reports, per benchmark and
per metric, how iteration 2 moves as the input it is handed is blended towards
the truth:

    alpha   iter1     iter2     improvement %
    0.0     ...       ...       ...            <- the stock run
    0.5     ...       ...       ...
    1.0     ...       ...       ...            <- pass 2 is handed the truth

ITER1 IS ONE NUMBER, NOT ONE PER ALPHA. The blend happens between pass 1 and
pass 2, so pass 1 cannot depend on alpha. That is asserted rather than assumed:
a disagreement means the patch reached pass 1 and every row here is suspect.

ALPHA > 0 IS LABEL LEAKAGE. Those rows are a diagnostic upper bound on what the
second pass can do with a better input, not a benchmark score. Only the
alpha = 0 row is comparable to anything published.

Usage
-----
    python build_table.py <results-dir>
    python build_table.py <results-dir> --iters 1 2
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("finar_exp004_table")

_REPEAT_RE = re.compile(r"^repeat(\d+)_(MASE|WQL)$")
_ALPHA_RE = re.compile(r"^alpha([0-9.]+)$")
METRICS = ("MASE", "WQL")

#: Columns that identify a scored row, all present ones used. fev needs
#: (task, target) because one task can be evaluated at two target
#: configurations; GIFT-Eval uses `dataset`. Requiring the result to be unique
#: is what makes this safe for both without hard-coding either.
KEY_CANDIDATES = ("task", "target", "dataset", "term", "horizon", "config")


def depth_columns(df: pd.DataFrame) -> dict[int, dict[str, str]]:
    cols: dict[int, dict[str, str]] = {}
    for c in df.columns:
        m = _REPEAT_RE.match(str(c))
        if m:
            cols.setdefault(int(m.group(1)), {})[m.group(2)] = c
    return cols


def join_key(df: pd.DataFrame, name: str) -> pd.Series:
    """One key per row, from every identifying column the csv has.

    NUL separator, which cannot occur in a task or dataset name, so the
    encoding is injective — "|" would map ("a|b","c") and ("a","b|c") together.
    """
    have = [c for c in KEY_CANDIDATES if c in df.columns]
    if not have:
        raise SystemExit(f"{name}: none of {KEY_CANDIDATES} is in the csv")
    key = df[have[0]].astype(str)
    for c in have[1:]:
        key = key + "\0" + df[c].astype(str)
    if key.duplicated().any():
        dup = key[key.duplicated()].iloc[0].replace("\0", " | ")
        raise SystemExit(
            f"{name}: {have} do not identify a row uniquely (e.g. {dup!r}); "
            f"the alphas cannot be aligned row for row")
    return key


def _win_rate(lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of rows where the deeper pass beats the shallower; ties half.

    Pairwise complete cases. The guard is not decoration: a row that is finite
    at one depth and not at the other would otherwise count as a loss.
    """
    both = np.isfinite(lo) & np.isfinite(hi)
    if not both.any():
        return float("nan")
    a, b = lo[both], hi[both]
    return float(((b < a) + 0.5 * (b == a)).mean())


def load(root: Path) -> dict[str, dict[float, pd.DataFrame]]:
    """``{benchmark: {alpha: frame}}`` for everything present under root."""
    out: dict[str, dict[float, pd.DataFrame]] = {}
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = {}
        for adir in sorted(bench_dir.iterdir()):
            m = _ALPHA_RE.match(adir.name)
            csv = adir / "results.csv"
            if m and csv.is_file():
                d = pd.read_csv(csv)
                d["_k"] = join_key(d, f"{bench_dir.name}/{adir.name}")
                runs[float(m.group(1))] = d.set_index("_k")
        if runs:
            out[bench_dir.name] = runs
    return out


def check_iter1_identical(runs: dict[float, pd.DataFrame], lo: int,
                          bench: str) -> None:
    """Pass 1 must not depend on alpha. Checked, because it is the axis.

    Raises rather than warns: the table would otherwise look entirely normal
    while its single `iter1` column was a fiction.
    """
    cols = {a: depth_columns(df).get(lo, {}).get("MASE")
            for a, df in runs.items()}
    have = [a for a, c in cols.items() if c]
    if len(have) < 2:
        return
    ref = have[0]
    a_ref = pd.to_numeric(runs[ref][cols[ref]], errors="coerce")
    for a in have[1:]:
        b = pd.to_numeric(runs[a][cols[a]], errors="coerce")
        common = a_ref.index.intersection(b.index)
        ok = np.isclose(a_ref.loc[common], b.loc[common],
                        rtol=1e-5, atol=1e-8, equal_nan=True)
        if not ok.all():
            raise SystemExit(
                f"{bench}: iteration {lo} differs between alpha={ref} and "
                f"alpha={a} on {int((~ok).sum())} of {len(common)} rows. The "
                f"blend must not touch pass {lo}.")
    logger.info("  %s: iteration %d identical across %d alphas",
                bench, lo, len(have))


def build(loaded: dict, lo: int, hi: int) -> pd.DataFrame:
    """One row per (benchmark, alpha, metric), on a population COMMON to all
    alphas — a row every alpha scored finitely at both depths.

    Scoring each alpha on its own survivors is not a comparison: an arm that
    happened to fail the hardest tasks would be measured on an easier
    population and look better for it.
    """
    rows = []
    for bench, runs in loaded.items():
        for metric in METRICS:
            usable = {}
            for a, d in runs.items():
                cols = depth_columns(d)
                if lo not in cols or hi not in cols:
                    continue
                if metric not in cols[lo] or metric not in cols[hi]:
                    continue
                x = pd.to_numeric(d[cols[lo][metric]], errors="coerce")
                y = pd.to_numeric(d[cols[hi][metric]], errors="coerce")
                ok = np.isfinite(x) & np.isfinite(y)
                usable[a] = (x[ok], y[ok])
            if not usable:
                continue
            common = sorted(set.intersection(
                *(set(x.index) for x, _ in usable.values())))
            if not common:
                continue
            for a in sorted(usable):
                x, y = usable[a]
                xx, yy = x.loc[common], y.loc[common]
                imp = (xx - yy) / xx.replace(0, np.nan) * 100.0
                rows.append({
                    "benchmark": bench, "alpha": a, "metric": metric,
                    "n": len(common),
                    f"iter{lo}": float(xx.mean()), f"iter{hi}": float(yy.mean()),
                    "gap": float(xx.mean() - yy.mean()),
                    "improvement_pct": float(imp.mean()),
                    "improvement_pct_sd": (float(imp.std(ddof=1))
                                           if len(imp) > 1 else np.nan),
                    "win_rate": _win_rate(xx.to_numpy(float),
                                          yy.to_numpy(float)),
                    "leakage": a > 0,
                })
    return pd.DataFrame(rows)


def render(t: pd.DataFrame, lo: int, hi: int) -> str:
    out = []
    for bench in sorted(t["benchmark"].unique()):
        for metric in METRICS:
            sub = t[(t["benchmark"] == bench) & (t["metric"] == metric)]
            if sub.empty:
                continue
            hdr = (f"{'alpha':>6}{'n':>5}{f'  iter{lo}':>10}{f'  iter{hi}':>10}"
                   f"{'improve%':>11}{'±sd':>8}{'win%':>7}")
            out += ["", f"── {bench} / {metric} " + "─" * 40, hdr,
                    "-" * len(hdr)]
            for _, r in sub.sort_values("alpha").iterrows():
                out.append(
                    f"{r['alpha']:>6.2f}{r['n']:>5.0f}{r[f'iter{lo}']:>10.4f}"
                    f"{r[f'iter{hi}']:>10.4f}{r['improvement_pct']:>+11.2f}"
                    f"{r.get('improvement_pct_sd', np.nan):>8.2f}"
                    f"{r['win_rate'] * 100:>7.1f}"
                    + ("   <- stock" if r["alpha"] == 0 else "   (leakage)"))
    out += ["",
            f"iter{lo} is alpha-independent by construction and asserted at "
            f"load: the blend",
            f"happens between pass {lo} and pass {hi}.",
            "improve% is (iter1 - iter2)/iter1 per row then averaged, so "
            "POSITIVE means the",
            "second pass helped. win% counts ties as half.",
            "EVERY alpha > 0 ROW IS LABEL LEAKAGE — a diagnostic upper bound, "
            "never a score."]
    return "\n".join(out)


def plot(t: pd.DataFrame, out_png: Path, lo: int, hi: int) -> None:
    """Improvement against alpha, one line per benchmark, one panel per metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in METRICS if not t[t["metric"] == m].empty]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 4.2),
                             squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for bench in sorted(t["benchmark"].unique()):
            sub = t[(t["benchmark"] == bench)
                    & (t["metric"] == metric)].sort_values("alpha")
            if sub.empty:
                continue
            ax.plot(sub["alpha"], sub["improvement_pct"], marker="o",
                    label=bench)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(r"$\alpha$   (0 = own forecast, 1 = ground truth)",
                      fontsize=9)
        ax.set_ylabel(f"{metric} improvement, iter{lo}→{hi} (%)", fontsize=10)
        ax.set_title(f"{metric}: what pass {hi} does with a better input",
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("alpha > 0 is label leakage — a diagnostic bound, not a score",
                 fontsize=8, y=0.01)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path)
    p.add_argument("--iters", nargs=2, type=int, default=[1, 2],
                   metavar=("LO", "HI"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    lo, hi = args.iters
    loaded = load(args.results)
    if not loaded:
        raise SystemExit(
            f"no results.csv under {args.results}/<benchmark>/alpha*/")
    for bench, runs in loaded.items():
        logger.info("%s: alphas %s", bench, sorted(runs))
        check_iter1_identical(runs, lo, bench)

    table = build(loaded, lo, hi)
    if table.empty:
        raise SystemExit(f"no rows: depths {lo}/{hi} missing from the csvs")
    dest = args.results / "exp004_table.csv"
    table.to_csv(dest, index=False)
    png = args.results / "exp004_improvement.png"
    plot(table, png, lo, hi)
    print(render(table, lo, hi))
    logger.info("\nwrote %s\nwrote %s", dest, png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
