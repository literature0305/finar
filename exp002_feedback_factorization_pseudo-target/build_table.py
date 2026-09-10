#!/usr/bin/env python3
"""Which half of the feedback earns the refinement — table and figure.

Reads the per-scenario csvs `run_eval.py` leaves under
``<results>/<benchmark>/<scenario>/results.csv`` and reports, per benchmark and
per metric:

                    iter1        iter2 full   iter2 self_only   iter2 cov_only
    fev_mul  MASE   ...          ...          ...               ...
    gift_mul MASE   ...          ...          ...               ...

`iter1` is ONE column, not three. The regimes only change what the write-back
hands to pass 2, so pass 1 is identical in all of them by construction; that is
asserted rather than assumed — if the three runs' `repeat1_MASE` disagree, the
patch leaked into pass 1 and every number here is suspect.

READING THE RESULT
`gap_pct` is ``(iter1 - iter2) / iter1``, so POSITIVE means the second pass
helped. The question the experiment asks is which of `self_only` and `cov_only`
recovers more of `full`'s improvement:

    gap_pct_self_only  ~ gap_pct_full   the target's own trajectory carries it
    gap_pct_cov_only   ~ gap_pct_full   the other variates carry it
    neither            ~ gap_pct_full   the two are complementary

ONE ASYMMETRY TO KEEP IN VIEW. `self_only` feeds back one variate and
`cov_only` feeds back n-1, so on wide groups they differ in how MUCH is fed
back as well as in what. The `mv_width_mean` column carries the mean group
width of the population each row was computed on, so a difference can be read
against it — and is BLANK when the results csv does not record a width.
GIFT-Eval writes only an `is_multivariate` boolean, so that block's width is
unknown here; `designation.json`, which `run_eval.py` writes beside the scores,
carries the true histogram.

UNIVARIATE TASKS ARE DEGENERATE HERE and are reported separately. With one
variate there is no "other": `self_only` feeds back everything (identical to
`full`) and `cov_only` feeds back nothing (identical to iteration 1). Pooling
them with the multivariate tasks would pull every regime toward the middle for
a reason that has nothing to do with the question. GIFT-Eval carries 54 such
tasks, so this is not a corner case there.

Usage
-----
    python build_table.py <results-dir>
    python build_table.py <results-dir> --iters 1 2
"""

from __future__ import annotations

import sys
import argparse
import ast
import logging
import re
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

logger = logging.getLogger("finar_exp002_pseudo_table")

_REPEAT_RE = re.compile(r"^repeat(\d+)_(MASE|WQL)$")

SCENARIOS = ("full", "self_only", "cov_only")
BENCHMARKS = ("fev_mul", "gift_mul")
METRICS = ("MASE", "WQL")
STRATA = ("multivariate", "univariate")

#: Columns that identify a scored row, tried in order and ALL of the present
#: ones used. fev needs (task, target) because a task evaluated at two target
#: configurations appears twice under one task name; GIFT-Eval uses `dataset`.
#: Taking whichever are present and then REQUIRING the result to be unique is
#: what makes this safe for both without hard-coding either — a schema that
#: does not identify a row uniquely raises instead of joining a row against an
#: arbitrary sibling.
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

    The separator is a NUL, which cannot occur in a task or dataset name, so
    the encoding is injective — joining with "|" would map ("a|b","c") and
    ("a","b|c") to the same key.
    """
    have = [c for c in KEY_CANDIDATES if c in df.columns]
    if not have:
        raise SystemExit(
            f"{name}: none of {KEY_CANDIDATES} is in the csv, so rows cannot "
            f"be aligned between scenarios")
    key = df[have[0]].astype(str)
    for c in have[1:]:
        key = key + "\0" + df[c].astype(str)
    if key.duplicated().any():
        dup = key[key.duplicated()].iloc[0].replace("\0", " | ")
        raise SystemExit(
            f"{name}: {have} do not identify a row uniquely (e.g. {dup!r}). "
            f"Joining the scenarios on this key would compare a row against an "
            f"arbitrary sibling.")
    return key


def n_variates(row) -> float:
    """Group width for one scored row, or NaN when the csv does not say.

    Three sources, because the two adapters answer differently: GIFT-Eval
    writes `is_multivariate` (a bool, so it gives the stratum but not the
    width), fev writes the target column list, and a plain count column is
    used if either ever grows one.
    """
    for c in ("n_vars", "n_variates", "num_variates"):
        v = row.get(c)
        if v is not None and pd.notna(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    tgt = row.get("target")
    if tgt is not None and pd.notna(tgt):
        try:
            parsed = ast.literal_eval(str(tgt))
            if isinstance(parsed, (list, tuple)):
                return float(len(parsed))
        except (ValueError, SyntaxError):
            pass
    mv = row.get("is_multivariate")
    if mv is not None and pd.notna(mv):
        # A BOOL, so it settles the stratum but not the width. Univariate is
        # exactly 1; multivariate is NaN, not 2. GIFT-Eval writes this flag as
        # a literal on every multivariate row and no count beside it, so
        # returning the smallest consistent value would print 2.0 as
        # `mv_width_mean` for the whole gift_mul block — a fabricated number in
        # the one column that exists to weigh `self_only` (1 variate) against
        # `cov_only` (width - 1). NaN says "not measured", which is true.
        return 1.0 if not bool(mv) else float("nan")
    return float("nan")


def stratum_from(row, n: float) -> str:
    """multivariate unless the row says otherwise.

    Separate from `n_variates` because the width can be unknown while the
    stratum is certain: `is_multivariate` gives the second without the first.
    """
    if np.isfinite(n):
        return "multivariate" if n >= 2 else "univariate"
    mv = row.get("is_multivariate")
    if mv is not None and pd.notna(mv):
        return "multivariate" if bool(mv) else "univariate"
    return "multivariate"      # both benchmarks are multivariate suites


def win_rate(lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of tasks where the deeper pass beats the shallower; ties half."""
    both = np.isfinite(lo) & np.isfinite(hi)
    if not both.any():
        return float("nan")
    a, b = lo[both], hi[both]
    return float(((b < a) + 0.5 * (b == a)).mean())


def load(root: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """``{benchmark: {scenario: frame}}`` for everything present under root."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for bench in BENCHMARKS:
        runs = {}
        for sc in SCENARIOS:
            csv = root / bench / sc / "results.csv"
            if csv.is_file():
                d = pd.read_csv(csv)
                d["_k"] = join_key(d, f"{bench}/{sc}")
                # ONE pass: `stratum_of` used to call `n_variates` again, so
                # every row parsed its target-column list twice.
                rows = [(r, n_variates(r)) for _, r in d.iterrows()]
                d["_nvar"] = [n for _, n in rows]
                d["_stratum"] = [stratum_from(r, n) for r, n in rows]
                runs[sc] = d.set_index("_k")
        if runs:
            out[bench] = runs
    return out


def check_iter1_identical(runs: dict[str, pd.DataFrame], lo: int,
                          bench: str) -> None:
    """Pass 1 must not depend on the regime. Checked, because it is the axis.

    The regimes change only what the write-back hands to pass 2. If pass 1
    differs between them the patch is reaching somewhere it should not, and the
    single `iter1` column this table prints would be a fiction — so this raises
    rather than warns: the table would otherwise look entirely normal.
    """
    cols = {sc: depth_columns(df).get(lo, {}).get("MASE")
            for sc, df in runs.items()}
    base = [sc for sc, c in cols.items() if c]
    if len(base) < 2:
        return
    base_sc = base[0]
    a = pd.to_numeric(runs[base_sc][cols[base_sc]], errors="coerce")
    for sc in base[1:]:
        b = pd.to_numeric(runs[sc][cols[sc]], errors="coerce")
        common = a.index.intersection(b.index)
        aa, bb = a.loc[common], b.loc[common]
        ok = np.isclose(aa, bb, rtol=1e-5, atol=1e-8, equal_nan=True)
        if not ok.all():
            bad = common[~ok][:3].tolist()
            raise SystemExit(
                f"{bench}: iteration {lo} differs between {base_sc!r} and "
                f"{sc!r} on {int((~ok).sum())} rows (e.g. {bad}). The regimes "
                f"must not touch pass {lo}; this raises rather than warns "
                f"because the table would otherwise look normal.")
        logger.info("  %s: iteration %d identical between %r and %r on %d rows",
                    bench, lo, base_sc, sc, int(ok.sum()))


def build(loaded: dict, lo: int, hi: int) -> pd.DataFrame:
    """One row per (benchmark, stratum, metric), on a COMMON population.

    Each scenario is filtered to the keys every scenario scored finitely at
    BOTH depths. Scoring each on its own surviving subset is not a comparison:
    a regime that happened to fail the hardest tasks would then be measured on
    an easier population and look better for it.
    """
    rows = []
    for bench, runs in loaded.items():
        for stratum in STRATA:
            for metric in METRICS:
                row = {"benchmark": bench, "stratum": stratum,
                       "metric": metric}
                usable = {}
                for sc, d in runs.items():
                    cols = depth_columns(d)
                    if lo not in cols or hi not in cols:
                        continue
                    if metric not in cols[lo] or metric not in cols[hi]:
                        continue
                    sub = d[d["_stratum"] == stratum]
                    if sub.empty:
                        continue
                    a = pd.to_numeric(sub[cols[lo][metric]], errors="coerce")
                    b = pd.to_numeric(sub[cols[hi][metric]], errors="coerce")
                    ok = np.isfinite(a) & np.isfinite(b)
                    usable[sc] = (a[ok], b[ok])
                if not usable:
                    continue
                common = set.intersection(
                    *(set(a.index) for a, _ in usable.values()))
                if not common:
                    continue
                common = sorted(common)
                row["n"] = len(common)
                dropped = max(len(a) - len(common) for a, _ in usable.values())
                if dropped:
                    row["n_dropped_to_align"] = dropped
                first = next(iter(usable))
                row[f"iter{lo}"] = float(usable[first][0].loc[common].mean())
                # The widths are a property of the POPULATION, not of a
                # scenario, so they are read once from the reference frame
                # rather than sliced for every arm and discarded for all but
                # one. NaN when the csv never said — see `n_variates`.
                w = runs[first]["_nvar"].loc[common]
                row["mv_width_mean"] = (float(w.mean()) if w.notna().any()
                                        else np.nan)
                for sc, (a, b) in usable.items():
                    aa, bb = a.loc[common], b.loc[common]
                    row[f"iter{hi}_{sc}"] = float(bb.mean())
                    row[f"gap_{sc}"] = float(aa.mean() - bb.mean())
                    # Denominator is THIS scenario's own iter1 on the common
                    # population, so numerator and denominator always agree.
                    row[f"gap_pct_{sc}"] = (row[f"gap_{sc}"] / aa.mean() * 100
                                            if aa.mean() else np.nan)
                    row[f"win_{sc}"] = win_rate(aa.to_numpy(float),
                                                bb.to_numpy(float))
                rows.append(row)
    return pd.DataFrame(rows)


def render(t: pd.DataFrame, lo: int, hi: int) -> str:
    out = []
    for metric in METRICS:
        sub = t[t["metric"] == metric]
        if sub.empty or f"iter{lo}" not in sub.columns:
            continue
        scs = [s for s in SCENARIOS if f"iter{hi}_{s}" in sub.columns]
        hdr = (f"{'benchmark':<10}{'stratum':<14}{'n':>5}{'width':>7}"
               f"{f'iter{lo}':>10}"
               + "".join(f"{f'{s} {hi}':>13}{'gap%':>8}{'win%':>7}" for s in scs))
        out += ["", f"── {metric} " + "─" * max(0, len(hdr) - len(metric) - 4),
                hdr, "-" * len(hdr)]
        for _, r in sub.iterrows():
            width = (f"{r['mv_width_mean']:>7.1f}"
                     if pd.notna(r.get("mv_width_mean")) else f"{'-':>7}")
            line = (f"{r['benchmark']:<10}{r['stratum']:<14}"
                    f"{r.get('n', float('nan')):>5.0f}" + width
                    + f"{r[f'iter{lo}']:>10.4f}")
            for s in scs:
                v, g, w = (r.get(f"iter{hi}_{s}"), r.get(f"gap_pct_{s}"),
                           r.get(f"win_{s}"))
                line += (f"{v:>13.4f}" if pd.notna(v) else f"{'-':>13}")
                line += (f"{g:>+8.2f}" if pd.notna(g) else f"{'-':>8}")
                line += (f"{w * 100:>7.1f}" if pd.notna(w) else f"{'-':>7}")
            out.append(line)
    out += ["",
            f"iter{lo} is the no-feedback baseline and is ONE column: the regimes",
            f"only change what pass {hi} is handed, so pass {lo} is identical in all",
            "three (asserted at load). gap% is (iter1 - iter2)/iter1, so POSITIVE",
            "means the second pass helped. win% counts ties as half.",
            "width is the mean group width of the population the row was computed",
            "on — self_only feeds back 1 variate and cov_only feeds back width-1,",
            "so read a difference between them against it. A blank width means",
            "the csv does not record one (GIFT-Eval writes only a bool); the true",
            "histogram is in designation.json beside the scores.",
            "univariate rows are degenerate: self_only == full and cov_only ==",
            f"iter{lo} there, because there is no other variate."]
    return "\n".join(out)


def plot(t: pd.DataFrame, out_png: Path, lo: int, hi: int) -> None:
    """Grouped bars: gap% per (benchmark, stratum), one bar per regime."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in METRICS if not t[t["metric"] == m].empty]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.4 * len(metrics), 4.4),
                             squeeze=False)
    colours = {"full": "#4C78A8", "self_only": "#F58518", "cov_only": "#54A24B"}
    for ax, metric in zip(axes[0], metrics):
        sub = t[t["metric"] == metric]
        labels = [f"{r['benchmark']}\n{r['stratum']}" for _, r in sub.iterrows()]
        scs = [s for s in SCENARIOS if f"gap_pct_{s}" in sub.columns]
        x = np.arange(len(sub))
        w = 0.8 / max(len(scs), 1)
        for i, s in enumerate(scs):
            ax.bar(x + i * w - 0.4 + w / 2, sub[f"gap_pct_{s}"].to_numpy(float),
                   width=w, label=s, color=colours.get(s))
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f"{metric} improvement, iter{lo}→{hi} (%)", fontsize=10)
        ax.set_title(f"{metric}: which feedback earns the refinement",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("positive = the second pass helped; univariate strata are "
                 "degenerate (no other variate)", fontsize=8, y=0.01)
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
            f"no results.csv under {args.results}/<benchmark>/<scenario>/")
    for bench, runs in loaded.items():
        logger.info("%s: %s", bench, ", ".join(f"{s} ({len(d)} rows)"
                                               for s, d in runs.items()))
        check_iter1_identical(runs, lo, bench)

    table = build(loaded, lo, hi)
    if table.empty:
        raise SystemExit(
            f"no rows: depths {lo}/{hi} missing from the csvs. Check for "
            f"repeat<d>_MASE / repeat<d>_WQL columns.")
    dest = args.results / "exp002_pseudo_table.csv"
    table.to_csv(dest, index=False)
    png = args.results / "exp002_pseudo_improvement.png"
    plot(table, png, lo, hi)
    print(render(table, lo, hi))
    logger.info("\nwrote %s\nwrote %s", dest, png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
