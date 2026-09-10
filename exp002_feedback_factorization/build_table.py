#!/usr/bin/env python3
"""exp002 tables — which half of the feedback earns the refinement.

Reads the three `run_eval.py` scenario runs and reports, per fev covariate
subset, what iteration 2 bought over iteration 1 under each regime:

              iter1        iter2 full   iter2 self_only   iter2 cov_only
    past_only  ...          ...          ...               ...
    future_known
    mixed

Iteration 1 is ONE column, not three: the regimes only change what the
write-back hands to pass 2, so pass 1 is identical in all of them. That is
asserted rather than assumed — if the three runs' `repeat1_MASE` disagree, the
patch leaked into pass 1 and the table says so instead of averaging it away.

SUBSETS. fev-bench records one `covariate` subset plus the raw
`known_dynamic_columns` / `past_dynamic_columns` lists, so the three-way split
is derived here from which of those lists is non-empty — the same rule
tsm-trainer's own `iteration_performance.py` uses, so the two agree.

WIN RATE is per task, iteration 2 against iteration 1, ties counting half —
again the repo's definition, so numbers from here mean what they mean there.

Usage
-----
    python build_table.py results/<model>
    python build_table.py results/<model> --iters 1 2
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

logger = logging.getLogger("finar_exp002_table")

SCENARIOS = ("full", "self_only", "cov_only")
SUBSETS = ("past_only", "future_known", "mixed")
METRICS = ("MASE", "WQL")
_REPEAT_RE = re.compile(r"^repeat(\d+)_(MASE|WQL)$")


def _nonempty_list(val) -> bool:
    """True when a csv cell holds a non-empty list.

    The columns arrive as strings ("['a', 'b']", "[]") after a round trip
    through csv, and occasionally as a real list when read from memory, so both
    are handled. A malformed cell counts as EMPTY rather than raising: the
    subset split is a reporting axis, and losing one task from a subset is
    better than losing the table.
    """
    if isinstance(val, (list, tuple)):
        return len(val) > 0
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "[]"):
        return False
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # RAISED, not guessed. This is the grouping key of the whole table: a
        # malformed cell read as "present" puts the task in the wrong subset
        # and read as "absent" drops it, and both are silent. An earlier
        # version returned bool(s) here, which classified "N/A" as a covariate
        # list.
        raise ValueError(
            f"cannot parse a dynamic-columns cell as a list: {val!r}") from None
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"dynamic-columns cell is not a list: {val!r}")
    return len(parsed) > 0


def subset_of(row) -> str | None:
    known = _nonempty_list(row.get("known_dynamic_columns", ""))
    past = _nonempty_list(row.get("past_dynamic_columns", ""))
    if known and past:
        return "mixed"
    if known:
        return "future_known"
    if past:
        return "past_only"
    return None


def depth_columns(df: pd.DataFrame) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for c in df.columns:
        m = _REPEAT_RE.match(str(c))
        if m:
            out.setdefault(int(m.group(1)), {})[m.group(2)] = c
    return out


def win_rate(lo: np.ndarray, hi: np.ndarray) -> tuple[float, int]:
    both = np.isfinite(lo) & np.isfinite(hi)
    n = int(both.sum())
    if n == 0:
        return float("nan"), 0
    a, b = lo[both], hi[both]
    return float(((b < a) + 0.5 * (b == a)).mean()), n


#: fev identifies a row by (task, target), NOT by task: a task evaluated at two
#: target configurations appears twice under one task name — the same collision
#: that forced the Seasonal-Naive baseline table to widen its key. Joining the
#: scenarios on `task` alone would compare a row against an arbitrary sibling.
JOIN_COLS = ("task", "target")


def join_key(df: pd.DataFrame) -> pd.Series:
    """One key per row, from ALL of JOIN_COLS.

    Every column is required rather than "whichever are present": falling back
    to `task` alone silently restores the defective join this exists to fix, on
    exactly the multi-target tasks where it breaks.

    The separator is a NUL, which cannot occur in a task or target name, so the
    encoding is injective — joining with "|" would map ("a|b","c") and
    ("a","b|c") to the same key.
    """
    missing = [c for c in JOIN_COLS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"csv is missing {missing}; the scenarios cannot be aligned row for "
            f"row on {JOIN_COLS}")
    key = df[JOIN_COLS[0]].astype(str)
    for c in JOIN_COLS[1:]:
        key = key + "\0" + df[c].astype(str)
    return key


def load(root: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for sc in SCENARIOS:
        csv = root / sc / "fev_bench.csv"
        if csv.is_file():
            df = pd.read_csv(csv)
            df["_subset"] = df.apply(subset_of, axis=1)
            out[sc] = df
    return out


def check_iter1_identical(runs: dict[str, pd.DataFrame], lo: int) -> None:
    """The regimes must not have touched pass 1."""
    cols = {sc: depth_columns(df).get(lo, {}).get("MASE") for sc, df in runs.items()}
    base_sc = "full" if "full" in runs else next(iter(runs))
    if not cols.get(base_sc):
        return
    def keyed(sc):
        df = runs[sc].copy()
        df["_k"] = join_key(df)
        dup = df["_k"].duplicated().sum()
        if dup:
            # Not a warning: with duplicate keys `.loc` returns an ambiguous
            # number of rows and the comparison silently pairs the wrong ones.
            raise SystemExit(
                f"{sc}: {dup} duplicate {JOIN_COLS} key(s) — the scenarios "
                f"cannot be aligned row for row. Example: "
                f"{df.loc[df['_k'].duplicated(keep=False), '_k'].iloc[0]!r}")
        return df.set_index("_k")[cols[sc]]

    base = keyed(base_sc)
    for sc in runs:
        if sc == base_sc or not cols.get(sc):
            continue
        other = keyed(sc)
        common = base.index.intersection(other.index)
        # Coverage first: a scenario that simply LOST the hard tasks would pass
        # an equality check computed only on what survived, and then look
        # better in the aggregate for having an easier population.
        missing = len(base.index.symmetric_difference(other.index))
        if missing:
            logger.warning("  %r and %r differ on %d task key(s) — the "
                           "populations are not the same", base_sc, sc, missing)
        a, b = base.loc[common].to_numpy(float), other.loc[common].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        n_bad = int((~ok).sum())
        if n_bad:
            logger.warning("  %r vs %r: %d common task(s) have a non-finite "
                           "iteration-%d score and were excluded from the check",
                           base_sc, sc, n_bad, lo)
        if ok.any() and not np.allclose(a[ok], b[ok], rtol=1e-6, atol=1e-9):
            n_diff = int((~np.isclose(a[ok], b[ok], rtol=1e-6, atol=1e-9)).sum())
            raise SystemExit(
                f"iteration {lo} differs between {base_sc!r} and {sc!r} on "
                f"{n_diff}/{int(ok.sum())} tasks (max |diff| "
                f"{float(np.nanmax(np.abs(a[ok] - b[ok]))):.3g}). Pass 1 cannot "
                f"depend on the feedback regime, so the patch has leaked into "
                f"it and the whole comparison is invalid. This is fatal rather "
                f"than a warning because the table would otherwise look normal.")
        else:
            logger.info("  iteration %d identical between %r and %r on %d "
                        "tasks", lo, base_sc, sc, int(ok.sum()))


def build(runs: dict[str, pd.DataFrame], lo: int, hi: int) -> pd.DataFrame:
    """One row per (subset, metric), scored on a population COMMON to all
    scenarios.

    Each scenario is filtered to the task keys that every scenario scored
    finitely at BOTH depths, and every number in the row — iter1, each
    scenario's iter2, its gap, its percentage and its win rate — comes from
    that one population.

    Scoring each scenario on its own surviving subset is what an earlier version
    did, and it is not a comparison: a regime that happened to fail the hardest
    tasks would then be measured on an easier population and look better for it.
    The percentage was worse still — its numerator came from the scenario and
    its denominator from whichever scenario supplied iter1 first, so the two
    halves of the ratio could describe different task sets.
    """
    # Index every run by key once.
    keyed = {}
    for sc, df in runs.items():
        d = df.copy()
        d["_k"] = join_key(d)
        keyed[sc] = d.set_index("_k")

    rows = []
    for subset in SUBSETS:
        for metric in METRICS:
            row = {"subset": subset, "metric": metric}
            usable = {}
            for sc, d in keyed.items():
                cols = depth_columns(d)
                if lo not in cols or hi not in cols:
                    continue
                if metric not in cols[lo] or metric not in cols[hi]:
                    continue
                sub = d[d["_subset"] == subset]
                if sub.empty:
                    continue
                a = pd.to_numeric(sub[cols[lo][metric]], errors="coerce")
                b = pd.to_numeric(sub[cols[hi][metric]], errors="coerce")
                ok = np.isfinite(a) & np.isfinite(b)
                usable[sc] = (a[ok], b[ok])
            if not usable:
                continue
            common = set.intersection(*(set(a.index) for a, _ in usable.values()))
            if not common:
                continue
            common = sorted(common)
            dropped = {sc: len(a) - len(common) for sc, (a, _) in usable.items()}
            row["n"] = len(common)
            if any(dropped.values()):
                row["n_dropped_to_align"] = max(dropped.values())
            first = next(iter(usable))
            row[f"iter{lo}"] = float(usable[first][0].loc[common].mean())
            for sc, (a, b) in usable.items():
                aa, bb = a.loc[common], b.loc[common]
                row[f"iter{hi}_{sc}"] = float(bb.mean())
                row[f"gap_{sc}"] = float(aa.mean() - bb.mean())
                # Denominator is THIS scenario's own iter1 on the common
                # population, so numerator and denominator always agree.
                row[f"gap_pct_{sc}"] = (row[f"gap_{sc}"] / aa.mean() * 100
                                        if aa.mean() else np.nan)
                row[f"win_{sc}"] = win_rate(aa.to_numpy(float),
                                            bb.to_numpy(float))[0]
            rows.append(row)
    return pd.DataFrame(rows)


def render(t: pd.DataFrame, lo: int, hi: int) -> str:
    out = []
    for metric in METRICS:
        sub = t[t["metric"] == metric]
        if sub.empty or f"iter{lo}" not in sub.columns:
            continue
        scs = [s for s in SCENARIOS if f"iter{hi}_{s}" in sub.columns]
        hdr = (f"{'subset':<14}{'n':>5}{f'iter{lo}':>10}" +
               "".join(f"{f'{s} {hi}':>13}{'gap%':>8}{'win%':>7}" for s in scs))
        out += ["", f"── {metric} " + "─" * max(0, len(hdr) - len(metric) - 4),
                hdr, "-" * len(hdr)]
        for _, r in sub.iterrows():
            line = f"{r['subset']:<14}{r.get('n', float('nan')):>5.0f}{r[f'iter{lo}']:>10.4f}"
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
            "means the second pass helped. win% counts ties as half."]
    return "\n".join(out)


def plot(t: pd.DataFrame, out_png: Path, lo: int, hi: int) -> None:
    """Grouped bars: gap% per subset, one bar per regime, one panel per metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.edgecolor": "#b0b0b0",
        "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    colours = {"full": "#2166ac", "self_only": "#f4a582", "cov_only": "#b2182b"}
    metrics = [m for m in METRICS if not t[t["metric"] == m].empty]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 3.4),
                             squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        sub = t[t["metric"] == metric].set_index("subset").reindex(SUBSETS)
        scs = [s for s in SCENARIOS if f"gap_pct_{s}" in sub.columns]
        x = np.arange(len(SUBSETS))
        w = 0.8 / max(len(scs), 1)
        for j, s in enumerate(scs):
            ax.bar(x + j * w - 0.4 + w / 2, sub[f"gap_pct_{s}"].to_numpy(float),
                   w, label=s.replace("_", "-"), color=colours.get(s), zorder=3)
        ax.axhline(0, color="#444", lw=0.8, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "-") for s in SUBSETS], fontsize=10)
        ax.set_ylabel(f"{metric} improvement, iter{lo}→{hi} (%)", fontsize=10)
        ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)
        ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
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
    runs = load(args.results)
    if not runs:
        raise SystemExit(f"no <scenario>/fev_bench.csv under {args.results}")
    logger.info("scenarios found: %s", ", ".join(runs))
    check_iter1_identical(runs, lo)

    table = build(runs, lo, hi)
    if table.empty:
        raise SystemExit(
            f"no rows: depths {lo}/{hi} missing, or no covariate tasks split "
            f"into subsets. Check the csv for repeat<d>_ columns.")
    table.to_csv(args.results / "exp002_table.csv", index=False)
    print(render(table, lo, hi))
    try:
        plot(table, args.results / "exp002_feedback.png", lo, hi)
    except Exception as e:
        logger.warning("plot failed: %s", e)
    logger.info("\nwrote %s/exp002_table.csv and .png", args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
