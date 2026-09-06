#!/usr/bin/env python3
"""Draw a 5,000-series subset of a training mixture, by tsm-trainer's own rule.

exp003 asks WHY iterative refinement helps — whether the second pass is
buying dependency modelling or acting as a regulariser. Answering that means
scoring a model on data drawn the way TRAINING draws it, not on a benchmark,
so this reproduces the trainer's sampling exactly and freezes the result as a
corpus that can be scored over and over.

=============================================================================
THE SAMPLING RULE, AS THE TRAINER IMPLEMENTS IT
=============================================================================
Copied in behaviour from `train_chronos2.Chronos2Dataset.__init__` (the block
that builds `_sample_probs`), not reinvented:

    weight_i     = data_points_i * custom_weight_i     (from the yaml entry)
    ds_prob_i    = weight_i / sum_j weight_j
    sample_prob  = ds_prob_i / n_i                     for each row of dataset i

So a dataset is chosen in proportion to its configured weight, and a row is
then drawn UNIFORMLY inside it. The weights come from tsm-trainer's own
`build_training_data_paths`, imported rather than re-parsed, so a change to
the entry format cannot make this file disagree with the trainer.

LENGTH FILTER FIRST, THEN RENORMALISE. Only rows of at least --min-length are
eligible, and `n_i` counts ELIGIBLE rows. A dataset with no eligible row is
dropped and its weight redistributed over the rest; leaving it in would leak
probability mass into a dataset that can never be drawn from, so the realised
mixture would not be the configured one. The dropped datasets and the shift in
each survivor's share are both reported.

=============================================================================
WHAT COMES OUT
=============================================================================
    <out-root>/train_subset_5k/            HF DatasetDict {"train": ...} with
                                           item_id / start / freq / target
    <out-root>/train_subset_5k_h48/        symlinks to the above, one per
    <out-root>/train_subset_5k_h160/       evaluation horizon — the evaluator
    <out-root>/train_subset_5k_h320/       names a result row after the
                                           DIRECTORY, so three horizons over
                                           one corpus need three names or the
                                           rows collide. Symlinks, not copies:
                                           it is the same data.
    <out-root>/train_subset_5k/manifest.json
                                           source yaml, seed, per-dataset draw
                                           counts, realised vs configured share

Usage
-----
    python build_subset.py --config <training yaml> --repo <tsm-trainer>
    python build_subset.py ... --n 5000 --min-length 500
    python build_subset.py ... --dry-run     # report the mixture, write nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("build_subset_5k")

DEFAULT_REPO = Path("/group-volume/workspace/mun-hak.lee/experiments/"
                    "tsm-trainer_001/tsm-trainer")

#: The evaluation horizons, in steps. 3 / 10 / 20 patches at input_patch_size 16.
HORIZONS = (48, 160, 320)


def add_repo_to_path(repo: Path) -> None:
    tr = repo / "scripts" / "forecasting" / "training"
    if not tr.is_dir():
        raise SystemExit(f"no training package under {repo} (looked at {tr})")
    for p in (str(tr), str(repo / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_mixture(config_path: Path, repo: Path):
    """``[(path, weight)]`` for every entry, using tsm-trainer's own parser."""
    import yaml

    add_repo_to_path(repo)
    from training_utils import build_training_data_paths  # noqa: E402

    cfg = yaml.safe_load(config_path.read_text())
    td = build_training_data_paths(cfg, lambda c: [])
    if td.dataset_weights is None:
        # Equal weights is what the trainer falls back to, so it is what this
        # must fall back to as well.
        logger.warning("%s carries no per-entry weights — using equal weights, "
                       "which is what the trainer does for such a config",
                       config_path.name)
        return [(p, 1.0) for p in td.paths]
    return list(zip(td.paths, td.dataset_weights))


def eligible_rows(path: str, min_length: int, target_col: str = "target"):
    """``(n_rows_total, indices of rows at least min_length long)``.

    OFFSETS ONLY — no value page is ever touched. An arrow list column stores
    its lengths as offsets beside the values, so `pc.list_value_length` answers
    "how long is each row" without decoding a single float.

    The obvious loop — `for i in range(len(chunk)): len(chunk[i][0])` — reads
    scalar by scalar and walks the whole memory-mapped file: measured at 16 GB
    of RSS on the 14 GB 100-variate corpus, before this function had even
    returned. That is file-backed and reclaimable rather than fatal on its own,
    but it is 14 GB of pointless page cache on a 23 GB machine, and it made the
    build look like it needed far more memory than it does.

    Two shapes to handle. A univariate corpus is `list<float>`, so the row
    length IS the outer length. A multivariate one is `list<list<float>>`,
    where the outer length is the VARIATE count and the time axis is the length
    of the first inner list — recovered from the flattened inner lengths at
    each row's starting offset.
    """
    import numpy as np
    import pyarrow.compute as pc
    from datasets import load_from_disk

    ds = load_from_disk(path)
    ds = ds["train"] if hasattr(ds, "keys") else ds
    if target_col not in ds.column_names:
        cand = [c for c in ds.column_names if c not in
                ("item_id", "start", "freq", "id", "timestamp")]
        if not cand:
            return len(ds), np.empty(0, dtype=np.int64)
        target_col = cand[0]

    # NOT combine_chunks(): concatenating the chunks of the 14 GB 100-variate
    # corpus into one array overflows arrow's int32 list offsets outright
    # ("offset overflow while concatenating arrays"), and copies 14 GB on the
    # way. `list_value_length` and `list_flatten` both accept a ChunkedArray
    # and preserve row order across chunks, so nothing needs combining.
    col = ds.data.table.column(target_col)
    outer = np.asarray(pc.list_value_length(col).to_numpy(zero_copy_only=False))
    inner_type = col.type.value_type
    if pa_is_list(inner_type):
        # 2-D: the time axis is the first inner list of each row.
        flat_len = np.asarray(pc.list_value_length(
            pc.list_flatten(col)).to_numpy(zero_copy_only=False))
        starts = np.concatenate([[0], np.cumsum(outer)[:-1]]).astype(np.int64)
        lengths = np.where(outer > 0, flat_len[np.clip(starts, 0,
                                                       len(flat_len) - 1)], 0)
    else:
        lengths = outer
    return len(ds), np.flatnonzero(lengths >= min_length).astype(np.int64)


def pa_is_list(t) -> bool:
    import pyarrow as pa
    return pa.types.is_list(t) or pa.types.is_large_list(t)


def _available_bytes() -> int:
    """MemAvailable, or 0 when it cannot be read (then the guards are inert)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _disk_free_bytes(path: Path) -> int:
    """Free bytes on the filesystem that will hold ``path``.

    Walks UP to the nearest existing directory: the output root is created
    later, and `disk_usage` on a path that does not exist yet raises, which the
    first version turned into "0 bytes free" and used to refuse a build that
    had 490 GB available.
    """
    import shutil
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return 0


def draw(mixture, min_length: int, n_draw: int, seed: int):
    """``(plan, report)`` — how many rows to take from each surviving dataset.

    `plan` is ``{path: sorted row indices}``; `report` records the configured
    and realised share of every entry so the redistribution is auditable.
    """
    rng = np.random.default_rng(seed)
    paths, weights = zip(*mixture)
    survivors, report = [], []

    for path, w in mixture:
        if not Path(path).exists():
            report.append({"path": path, "status": "missing", "weight": w})
            continue
        try:
            total, idx = eligible_rows(path, min_length)
        except Exception as e:
            # LOUD. Swallowing this silently dropped the two corpora carrying
            # 93% of the sampling weight — an ArrowInvalid from a since-removed
            # combine_chunks() — and the build happily produced a "training
            # subset" drawn from the remaining 7%. A mixture that cannot be
            # read as configured is not a mixture worth sampling.
            raise SystemExit(
                f"cannot read {path} (share "
                f"{w / sum(x[1] for x in mixture):.4f} of the mixture): "
                f"{type(e).__name__}: {e}") from e
        if len(idx) == 0:
            report.append({"path": path, "status": "no row >= min_length",
                           "weight": w, "n_rows": total})
            continue
        survivors.append((path, float(w), total, idx))

    if not survivors:
        raise SystemExit(f"no dataset has a row of at least {min_length} steps")

    w_total = sum(w for _, w, _, _ in survivors)
    cfg_total = sum(weights)
    # One multinomial over datasets, then uniform inside each — the two-stage
    # rule written as one draw, which is what makes the per-row probability
    # ds_prob/n exactly the trainer's.
    ds_probs = np.array([w / w_total for _, w, _, _ in survivors])
    counts = rng.multinomial(n_draw, ds_probs)

    plan = {}
    for (path, w, total, idx), take in zip(survivors, counts):
        take = int(min(take, len(idx)))    # cannot draw more rows than exist
        if take:
            plan[path] = np.sort(rng.choice(idx, size=take, replace=False))
        # EVERY survivor is reported, including take == 0. Reporting only the
        # ones that won rows made "eligible but not drawn" indistinguishable
        # from "never considered" in the manifest — 19 of 37 entries appeared,
        # and the missing 18 were the small-share datasets the multinomial
        # happened to give nothing, which is exactly what an audit needs to
        # show.
        report.append({
            "path": path, "status": "ok" if take else "eligible, drew 0",
            "weight": w, "n_rows": total, "n_eligible": int(len(idx)),
            "n_drawn": take,
            "configured_share": w / cfg_total,
            "renormalised_share": w / w_total,
        })
    return plan, report


#: Rows converted to Python at a time. A row of the 100-variate x 8192 corpus
#: is 819,200 floats ~ 26 MB as Python objects, so this caps the conversion at
#: ~0.4 GB however wide the corpus is.
BATCH = 16


def _row_bytes(path: str) -> int:
    """Bytes per row on disk, from the arrow files — no rows are read."""
    files = list(Path(path).rglob("*.arrow"))
    n_bytes = sum(f.stat().st_size for f in files)
    from datasets import load_from_disk
    ds = load_from_disk(path)
    ds = ds["train"] if hasattr(ds, "keys") else ds
    return int(n_bytes / max(len(ds), 1))


def estimate(plan) -> dict:
    """Output size and the peak Python conversion, BEFORE anything is built.

    The first version of this file died here, and took the whole WSL VM with
    it: it wrote `"target": sub[tgt]`, and `Dataset.__getitem__` converts a
    whole arrow column to Python lists. With 2,197 rows drawn from the
    100-variate x 8192 corpus that is 1.8e9 floats — 58 GB of Python objects,
    then `from_dict` rebuilding arrow beside it, against a 23 GB machine. The
    allocation overshoots so far that the kernel cannot OOM-kill its way out
    and the VM goes down.

    Nothing here reads a row: the per-row size comes from the arrow file sizes
    divided by the row count. The estimate is reported and enforced so a
    mixture that would not fit stops before it starts rather than after.
    """
    per_row = {p: _row_bytes(p) for p in plan}
    out_bytes = sum(per_row[p] * len(i) for p, i in plan.items())
    peak_py = max(per_row.values(), default=0) * BATCH * 4   # arrow -> Python
    return {"out_bytes": out_bytes, "peak_python_bytes": peak_py,
            "per_row": per_row}


def materialise(plan, out_dir: Path, freq_default: str = "h"):
    """Concatenate the drawn rows into one HF dataset, STAYING IN ARROW.

    The target column is never converted as a whole. `select` is lazy, and the
    schema normalisation runs through `map(batched=True, batch_size=BATCH)`, so
    at most BATCH rows exist as Python objects at any moment. The small columns
    (item_id / start / freq are short strings) are cheap and are read directly.

    The normalisation also has to happen: the corpora disagree about whether a
    target is 1-D (univariate, `list<float>`) or 2-D (`list<list<float>>`), and
    `concatenate_datasets` refuses a schema mismatch. Everything is widened to
    2-D, so one univariate series becomes a 1 x T row.
    """
    import datasets as hf

    parts = []
    for path, idx in plan.items():
        ds = hf.load_from_disk(path)
        ds = ds["train"] if hasattr(ds, "keys") else ds
        sub = ds.select(idx.tolist())
        cols = sub.column_names
        tgt = "target" if "target" in cols else next(
            c for c in cols if c not in ("item_id", "start", "freq", "id",
                                         "timestamp"))
        # Short strings: safe to read as a column, unlike the target.
        starts = ([str(x) for x in sub["start"]] if "start" in cols
                  else ["2020-01-01T00:00:00"] * len(sub))
        freqs = ([str(x) for x in sub["freq"]] if "freq" in cols
                 else [freq_default] * len(sub))

        def widen(batch, _t=tgt):
            out = []
            for v in batch[_t]:
                # A 2-D row's first element is itself a sequence; a 1-D row's
                # is a number. Wrapping the 1-D case keeps one schema.
                out.append(v if (len(v) and isinstance(v[0], (list, tuple)))
                           else [v])
            return {"target": out}

        sub = sub.map(widen, batched=True, batch_size=BATCH,
                      writer_batch_size=BATCH, remove_columns=cols,
                      desc=Path(path).name[:32], load_from_cache_file=False)
        name = Path(path).name
        sub = sub.add_column("item_id", [f"{name}__{i}" for i in idx.tolist()])
        sub = sub.add_column("start", starts)
        sub = sub.add_column("freq", freqs)
        parts.append(sub)

    merged = hf.concatenate_datasets(parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf.DatasetDict({"train": merged}).save_to_disk(str(out_dir))
    return len(merged)


def link_horizons(base: Path, horizons=HORIZONS) -> list[Path]:
    """One symlink per horizon beside the corpus.

    The evaluator names a result row after the DIRECTORY, so listing one
    corpus three times under three prediction_lengths would produce three rows
    with the same name — indistinguishable downstream, and the pairwise tools
    reject a csv with duplicate keys outright. Symlinks give three names over
    one copy of the data.
    """
    made = []
    for h in horizons:
        link = base.parent / f"{base.name}_h{h}"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(base, target_is_directory=True)
        made.append(link)
    return made


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True,
                   help="training yaml whose training_data defines the mixture")
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO,
                   help="tsm-trainer checkout to IMPORT the parser from")
    p.add_argument("--out-root", type=Path, default=Path("/group-volume/ts-dataset"))
    p.add_argument("--name", default="train_subset_5k")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--min-length", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    mixture = load_mixture(args.config, args.repo)
    logger.info("mixture: %d entries from %s", len(mixture), args.config.name)

    plan, report = draw(mixture, args.min_length, args.n, args.seed)
    drawn = sum(len(v) for v in plan.values())
    ok = [r for r in report if r["status"] == "ok"]
    skipped = [r for r in report if r["status"] != "ok"]
    logger.info("drew %d rows (>= %d steps) from %d/%d datasets",
                drawn, args.min_length, len(ok), len(mixture))
    for r in sorted(ok, key=lambda r: -r["n_drawn"])[:8]:
        logger.info("  %-52s n=%-5d cfg=%.4f -> %.4f",
                    Path(r["path"]).name[:52], r["n_drawn"],
                    r["configured_share"], r["renormalised_share"])
    if skipped:
        logger.info("  skipped %d entr(ies): %s", len(skipped),
                    ", ".join(sorted({r["status"].split(":")[0] for r in skipped})))
    if drawn < args.n:
        logger.warning("only %d of the requested %d rows are available at "
                       "min-length %d", drawn, args.n, args.min_length)
    # PRE-FLIGHT, before a single row is written. The first version of this
    # file overshot a 23 GB machine by 3x and took the WSL VM down with it, so
    # the size is now computed from arrow file sizes and checked against what
    # the host actually has free.
    est = estimate(plan)
    free = _available_bytes()
    disk_free = _disk_free_bytes(args.out_root)
    logger.info("estimate: output %.1f GB, peak Python %.2f GB "
                "(RAM available %.1f GB, disk free %.1f GB)",
                est["out_bytes"] / 1e9, est["peak_python_bytes"] / 1e9,
                free / 1e9, disk_free / 1e9)
    if est["peak_python_bytes"] > 0.5 * free:
        raise SystemExit(
            f"peak conversion {est['peak_python_bytes']/1e9:.1f} GB is more "
            f"than half the {free/1e9:.1f} GB available — lower --n or BATCH")
    if est["out_bytes"] > 0.9 * disk_free:
        raise SystemExit(
            f"output {est['out_bytes']/1e9:.1f} GB does not fit in "
            f"{disk_free/1e9:.1f} GB free at {args.out_root}")

    if args.dry_run:
        return 0

    out = args.out_root / args.name
    n = materialise(plan, out)
    links = link_horizons(out)
    (out / "manifest.json").write_text(json.dumps({
        "source_config": str(args.config), "seed": args.seed,
        "n_requested": args.n, "n_written": n, "min_length": args.min_length,
        "horizons": list(HORIZONS), "entries": report,
    }, indent=1, default=str))
    logger.info("wrote %d series to %s", n, out)
    logger.info("horizon links: %s", ", ".join(x.name for x in links))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
