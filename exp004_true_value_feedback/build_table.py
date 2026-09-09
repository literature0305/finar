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
import json
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


def run_depth(root: Path) -> int:
    """The depth the run was scored at, from its own manifest.

    Read rather than re-supplied. `run_exp004.sh --stage table` used to pass its
    own `--depth`, which defaults to 2: scoring at depth 16 and then building
    the table without repeating `--depth 16` tabled iterations 1 and 2 and threw
    the other fourteen away, with no error. The number is already on disk.
    """
    man = root / "manifest.json"
    if not man.is_file():
        raise SystemExit(
            f"{man} is missing, so the depth this run was scored at is unknown. "
            f"Pass --iters LO HI explicitly.")
    d = int(json.loads(man.read_text()).get("coe_eval_depth", 0))
    if d < 2:
        raise SystemExit(f"{man} records coe_eval_depth={d}; nothing to contrast")
    return d


def alpha_iteration_tables(loaded: dict, depth: int) -> pd.DataFrame:
    """MASE and task-level win rate for every (benchmark, alpha, iteration).

    ONE TASK POPULATION FOR THE WHOLE SURFACE, per benchmark. Averaging each
    cell over whatever happened to be finite in it makes the alpha and iteration
    axes compare different task sets — the alpha=1 column could be a mean over
    tasks the alpha=0 column never scored, and the difference would read as an
    effect. The population here is the tasks present for EVERY alpha and finite
    at EVERY iteration, so `n_tasks` describes every cell and the surface is
    internally comparable. It is also, for the same reason, not necessarily the
    population `build()` reports on: that one restricts to two depths, this one
    to all of them. The two tables answer different questions and the counts say
    so.

    EVERY DEPTH MUST BE PRESENT. `manifest.json` records the depth the run was
    ASKED for, not proof that the adapter emitted it; an interrupted run would
    otherwise produce a sparse surface that looks like a valid experiment with
    fewer columns.

    THE WIN RATE IS OVER TASKS, NOT ITEMS, and the column says so. Per-item
    would be the better statistic and is not available: fev-bench scores its
    repeat sets through `_compute_metrics_direct` and GIFT-Eval through
    `score_and_plot_repeats`, the two share no seam, and the second discards
    `mase_per_item` before returning. Reaching item level means two hooks into
    read-only internals, one of which that file already documents as fragile.

    `blended_MASE` is `(1 - alpha) * MASE`, the MASE of the intermediate the
    next pass is handed. It is an ALGEBRAIC IDENTITY, not a measurement: the
    blend is `(1-a)*p + a*y` and MASE is `mean|y - .|/s`, so
    `mean|y - ((1-a)p + a*y)| = (1-a)*mean|y - p|` exactly, for `a` in [0, 1].
    It cannot fail — a check of whether the blend reached the model is
    `truth_hits` in truth.json, not this column.
    """
    rows = []
    for bench, runs in loaded.items():
        percols, missing = {}, []
        for alpha, df in sorted(runs.items()):
            percols[alpha] = depth_columns(df)
            miss = [i for i in range(1, depth + 1)
                    if not percols[alpha].get(i, {}).get("MASE")]
            if miss:
                missing.append(f"alpha={alpha:g} missing depths {miss}")
        if missing:
            raise SystemExit(
                f"{bench}: the run was scored to depth {depth} but its csvs do "
                f"not carry every depth — " + "; ".join(missing) + ". A sparse "
                f"surface would look like a valid experiment with fewer "
                f"columns, so it is refused.")

        idx = None
        for df in runs.values():
            idx = df.index if idx is None else idx.intersection(df.index)
        ok = pd.Series(True, index=idx)
        for alpha, df in runs.items():
            for i in range(1, depth + 1):
                v = pd.to_numeric(df.loc[idx, percols[alpha][i]["MASE"]],
                                  errors="coerce")
                ok &= np.isfinite(v)
        keep = ok[ok].index
        if not len(keep):
            raise SystemExit(
                f"{bench}: no task is finite at every one of depths "
                f"1..{depth} for every alpha, so there is no population the "
                f"surface could be computed on")
        if len(keep) < len(idx):
            logger.warning(
                "  %s: %d of %d tasks dropped — not finite at every depth for "
                "every alpha", bench, len(idx) - len(keep), len(idx))

        for alpha, df in sorted(runs.items()):
            b = pd.to_numeric(df.loc[keep, percols[alpha][1]["MASE"]],
                              errors="coerce").to_numpy(float)
            for it in range(1, depth + 1):
                v = pd.to_numeric(df.loc[keep, percols[alpha][it]["MASE"]],
                                  errors="coerce").to_numpy(float)
                mase = float(v.mean())
                rows.append({
                    "benchmark": bench, "alpha": alpha, "iteration": it,
                    "MASE": mase,
                    "blended_MASE": (1.0 - alpha) * mase,
                    # 0.5, not 1.0, at it == 1: iteration 1 against itself ties
                    # on every task and `_win_rate` scores ties at a half.
                    "win_rate_tasks_vs_iter1": _win_rate(b, v),
                    "n_tasks": len(keep),
                })
    return pd.DataFrame(rows)


def alpha_iteration_heatmaps(t: pd.DataFrame, dest: Path) -> None:
    """One row of panels per benchmark: MASE, then win rate. x = iteration,
    y = alpha, as asked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benches = sorted(t["benchmark"].unique())
    panels = [("MASE", "viridis_r", "MASE (lower is better)"),
              ("win_rate_tasks_vs_iter1", "RdBu",
               "task win rate vs iteration 1")]
    fig, axes = plt.subplots(len(benches), 2, squeeze=False,
                             figsize=(12, 3.6 * len(benches)))
    for row, bench in enumerate(benches):
        sub = t[t["benchmark"] == bench]
        for col, (value, cmap, title) in enumerate(panels):
            ax = axes[row][col]
            # pivot, not pivot_table: one row per (alpha, iteration) is the
            # contract above, so a duplicate must raise here rather than be
            # quietly averaged into something that looks fine.
            piv = sub.pivot(index="alpha", columns="iteration", values=value)
            kw = {"vmin": 0.0, "vmax": 1.0} if col else {}
            im = ax.imshow(piv.values, aspect="auto", cmap=cmap, **kw)
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels(piv.columns, fontsize=7)
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels([f"{a:g}" for a in piv.index])
            ax.set_xlabel("iteration"); ax.set_ylabel("alpha")
            ax.set_title(f"{bench}: {title}", fontsize=10)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                                fontsize=6,
                                color="k" if col else "w")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("alpha = how much of the TRUE horizon pass n was handed; "
                 "alpha > 0 is label leakage and an upper bound, not a score",
                 fontsize=9, y=0.005)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    plt.close(fig)


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
    p.add_argument("--iters", nargs=2, type=int, default=None,
                   metavar=("LO", "HI"),
                   help="depths to contrast. Default: 1 and the depth the run "
                        "was actually scored at, read from manifest.json — "
                        "passing a HI the run never reached silently tables a "
                        "shallower experiment than the one that ran.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    lo, hi = args.iters if args.iters else (1, run_depth(args.results))
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

    # run_depth(), not `hi`: `--iters 1 2` selects the CONTRAST pair, and
    # feeding it here would silently shrink a depth-16 run's surface to two
    # columns — the failure run_depth() exists to prevent, one caller over.
    ai = alpha_iteration_tables(loaded, run_depth(args.results))
    ai_csv = args.results / "exp004_alpha_iteration.csv"
    ai.to_csv(ai_csv, index=False)
    ai_png = args.results / "exp004_alpha_iteration.png"
    alpha_iteration_heatmaps(ai, ai_png)
    for bench in sorted(ai["benchmark"].unique()):
        sub = ai[ai["benchmark"] == bench]
        piv = sub.pivot_table(index="alpha", columns="iteration", values="MASE")
        print(f"\n── {bench}: MASE by alpha x iteration "
              f"(n={int(sub['n_tasks'].max())} tasks) ──")
        print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    logger.info("\nwrote %s\nwrote %s\nwrote %s\nwrote %s",
                dest, png, ai_csv, ai_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
