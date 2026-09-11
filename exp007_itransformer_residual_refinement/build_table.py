#!/usr/bin/env python3
"""Collect exp007's runs into one table and one figure.

Reads every `metrics.json` under a results root and answers the experiment's
question directly: for each (dataset, horizon), what did the baseline score,
what did the refinement score, and did the refinement help. The published cell
is carried alongside, so a comparison against a baseline that did NOT reproduce
the paper is visible as such rather than quietly presented as a result.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(next(d for d in Path(__file__).resolve().parents
                         if (d / "finar_cpu.py").is_file())))
from finar_cpu import limit_cpu  # noqa: E402

limit_cpu(quiet=True)

import paper  # noqa: E402
import run_eval  # noqa: E402
from itransformer import ModelConfig  # noqa: E402

logger = logging.getLogger("finar_exp007")

FIELDS = ("dataset", "pred_len", "variant", "depth", "headline", "mse",
          "mae", "paper_mse", "paper_mae", "verdict", "protocol", "run")


def collect(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        want = paper.target(m["dataset"], m["pred_len"]) or (None, None)
        # The protocol is derived from the run's OWN config.json, not read back
        # out of metrics.json: a run written before the fingerprint existed has
        # no digest there, and a missing digest defaulting to "" made every
        # such run match every other one — exactly the comparison the
        # fingerprint exists to refuse.
        protocol = _protocol_of_run(path.parent)
        digest = run_eval.protocol_digest(protocol)
        headline_depth = int(m.get("headline_depth",
                                   max(int(d) for d in m["by_depth"])))
        for depth, score in sorted(m["by_depth"].items(),
                                   key=lambda kv: int(kv[0])):
            is_headline = int(depth) == headline_depth
            rows.append({
                "dataset": m["dataset"], "pred_len": m["pred_len"],
                "variant": m["variant"], "depth": int(depth),
                "protocol": digest, "_protocol": protocol,
                "mse": round(score["mse"], 6), "mae": round(score["mae"], 6),
                "paper_mse": want[0], "paper_mae": want[1],
                # The verdict belongs to the run's HEADLINE depth — the one the
                # checkpoint is configured to run. A sweep's other depths are
                # measurements around it, not what it claims to be.
                "verdict": m["paper_verdict"] if is_headline else "",
                "headline": is_headline,
                "run": m["run"],
            })
    return rows


def _protocol_of_run(run_dir: Path) -> dict:
    """The comparison fingerprint of a run directory, from its `config.json`."""
    meta = json.loads((run_dir / "config.json").read_text())
    return run_eval.protocol_of(meta, ModelConfig.from_dict(meta["model"]))


def compare(rows: list[dict]) -> list[dict]:
    """Baseline vs each refinement variant, per (dataset, horizon).

    A pair is only formed when the two runs share a PROTOCOL — lookback, seed,
    optimizer settings, model dimensions, data root (`run_eval.PROTOCOL_FIELDS`).
    Flipping `--refinement` changes none of those, so a legitimate pair always
    matches; what does not match is a refinement measured against a baseline
    from another session with a different seed or learning rate, and that
    comparison is REFUSED rather than footnoted, because its gain is not a gain.

    TWO gains are reported for a matched pair, because they answer different
    questions and only one of them is a result:

      `improvement_pct` — at the depth the checkpoint is CONFIGURED to run.
        This is the honest number: the depth was chosen before the test split
        was scored.
      `improvement_pct_best_depth` — at whichever swept depth scored best.
        That is selection on the test set, so it is an upper bound on what
        picking a depth could buy, not a score.
    """
    # LISTS, not last-one-wins. Two runs with the same key are two runs; a dict
    # would keep whichever `collect` happened to read last and report it as
    # "the" baseline, which is the silent-selection failure this function
    # exists to avoid.
    headline: dict = {}
    best_at: dict = {}
    for r in rows:
        key = (r["dataset"], r["pred_len"], r["variant"], r["protocol"])
        if r.get("headline"):
            headline.setdefault(key, []).append(r)
        if key not in best_at or r["mse"] < best_at[key]["mse"]:
            best_at[key] = r
    baselines: dict = {}
    for key, group in headline.items():
        if key[2] == "baseline":
            baselines.setdefault((key[0], key[1]), []).extend(group)

    out = []
    for key in sorted(headline):
        dataset, horizon, variant, digest = key
        if variant == "baseline":
            continue
        group, best = headline[key], best_at[key]
        record = {
            "dataset": dataset, "pred_len": horizon, "variant": variant,
            "protocol": digest, "baseline_mse": None, "refined_mse": None,
            "improvement_pct": None, "eval_depth": None,
            "refined_mse_best_depth": None, "improvement_pct_best_depth": None,
            "best_depth": None, "note": "",
        }
        if len(group) > 1:
            record["note"] = (
                f"{len(group)} runs of this variant share a protocol "
                f"({', '.join(sorted(r['run'] for r in group))}) — ambiguous, "
                f"not compared")
            out.append(record)
            continue
        row = group[0]
        record["refined_mse"] = row["mse"]
        record["eval_depth"] = row["depth"]
        record["refined_mse_best_depth"] = best["mse"]
        record["best_depth"] = best["depth"]
        pool = baselines.get((dataset, horizon), [])
        matched = [b for b in pool if b["protocol"] == digest]
        if len(matched) > 1:
            record["note"] = (
                f"{len(matched)} baseline runs share this protocol "
                f"({', '.join(sorted(b['run'] for b in matched))}) — "
                f"ambiguous, not compared")
            out.append(record)
            continue
        if not matched:
            differs = ""
            if pool:
                other = pool[0].get("_protocol", {})
                mine = row.get("_protocol", {})
                fields = sorted(k for k in set(other) | set(mine)
                                if other.get(k) != mine.get(k))
                differs = (f" (nearest, {pool[0]['run']}, differs in "
                           f"{', '.join(fields)})" if fields else "")
            record["note"] = ("no baseline run shares this run's protocol"
                              + differs + " — not compared")
            out.append(record)
            continue
        base = matched[0]
        record["baseline_mse"] = round(base["mse"], 6)
        record["refined_mse"] = round(row["mse"], 6)
        record["refined_mse_best_depth"] = round(best["mse"], 6)
        record["improvement_pct"] = round(
            100.0 * (base["mse"] - row["mse"]) / base["mse"], 3)
        record["improvement_pct_best_depth"] = round(
            100.0 * (base["mse"] - best["mse"]) / base["mse"], 3)
        if base["verdict"] == "miss":
            record["note"] = (
                "the baseline missed its published cell — read the gain as "
                "relative to THIS baseline, not to the paper's")
        out.append(record)
    return out


def figure(rows: list[dict], comparisons: list[dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed — table only, no figure")
        return False
    if not comparisons:
        logger.info("no baseline/refinement pair yet — no figure")
        return False

    depth_rows = [r for r in rows if r["variant"] != "baseline"]
    sweeps = {}
    for r in depth_rows:
        sweeps.setdefault((r["dataset"], r["pred_len"], r["variant"]), []).append(r)
    sweeps = {k: sorted(v, key=lambda r: r["depth"])
              for k, v in sweeps.items() if len(v) > 1}

    ncols = 2 if sweeps else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4.5), squeeze=False)
    ax = axes[0][0]
    # Only the pairs that were actually formed; an uncomparable row has no
    # gain, and plotting it as 0 would read as "no effect".
    comparisons = [c for c in comparisons if c["improvement_pct"] is not None]
    if not comparisons:
        logger.info("no comparable baseline/refinement pair — no figure")
        return False
    labels = [f"{c['dataset']}\n{c['pred_len']}" for c in comparisons]
    gains = [c["improvement_pct"] for c in comparisons]
    colors = ["#2b7a3d" if g > 0 else "#a33" for g in gains]
    ax.bar(range(len(gains)), gains, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MSE improvement over baseline (%)")
    ax.set_title("exp007: residual refinement vs the paper's iTransformer")

    if sweeps:
        ax = axes[0][1]
        for (dataset, horizon, variant), rs in sorted(sweeps.items()):
            base = next((c["baseline_mse"] for c in comparisons
                         if c["dataset"] == dataset
                         and c["pred_len"] == horizon), None)
            ys = [100.0 * (base - r["mse"]) / base if base else r["mse"]
                  for r in rs]
            ax.plot([r["depth"] for r in rs], ys, marker="o",
                    label=f"{dataset}/{horizon} {variant}")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("chain depth at inference (coe_eval_depth)")
        ax.set_ylabel("MSE improvement over baseline (%)")
        ax.set_title("test-time scaling")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", help="directory holding the run directories")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        datefmt="%m-%d %H:%M:%S")
    root = Path(args.root)
    rows = collect(root)
    if not rows:
        logger.warning("no metrics.json under %s — nothing to table", root)
        return
    table = root / "exp007_table.csv"
    with open(table, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("%d row(s) -> %s", len(rows), table)

    comparisons = compare(rows)
    if comparisons:
        path = root / "exp007_improvement.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(comparisons[0]),
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(comparisons)
        logger.info("%d comparison(s) -> %s", len(comparisons), path)
        for c in comparisons:
            logger.info("  %s/%s %s: baseline %s -> %s at depth %s (%s%%); "
                        "best swept depth %s gives %s (%s%%, test-set "
                        "selection) %s",
                        c["dataset"], c["pred_len"], c["variant"],
                        c["baseline_mse"], c["refined_mse"], c["eval_depth"],
                        c["improvement_pct"], c["best_depth"],
                        c["refined_mse_best_depth"],
                        c["improvement_pct_best_depth"], c["note"])
    if figure(rows, comparisons, root / "exp007_refinement.png"):
        logger.info("figure -> %s", root / "exp007_refinement.png")

    missed = sorted({f"{r['dataset']}/{r['pred_len']}" for r in rows
                     if r["variant"] == "baseline" and r["verdict"] == "miss"})
    if missed:
        logger.warning("baseline(s) outside the published tolerance: %s — the "
                       "refinement's gain is measured against these, so say so "
                       "when reporting it", ", ".join(missed))


if __name__ == "__main__":
    main()
