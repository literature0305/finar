#!/usr/bin/env python3
"""Re-cut the cross-horizon length sweep so every horizon shows the model the
SAME amount of observed history.

WHY THIS EXISTS
---------------
The published corpus (`/group-volume/ts-dataset/cross_horizon_length`) holds the
observed context byte-identical across all twelve subsets — 8192 series, 2048
steps, same values in every corpus and regime. The MODEL does not see 2048 of
them. `aed/model_base.py::append_forecast_region` carves the forecast region out
of the context window:

    forecast_len = ceil(prediction_length / patch_size) * patch_size
    max_ctx      = context_length - forecast_len
    context      = context[..., -max_ctx:]

so with `context_length=2048, patch_size=16` the history actually reaching the
encoder is

    H=16   -> 2032      H=100 -> 1936      H=400 -> 1648      H=1000 -> 1040

The horizon axis is therefore collinear with "how much history the model got",
and a MASE that rises with H cannot be attributed to the horizon. This script
removes that confound the only way it can be removed — by making the stored
context short enough that no horizon truncates it — so the sweep varies the
horizon and nothing else.

1040, NOT 1048. `forecast_len` is rounded UP to a patch multiple:
ceil(1000/16)*16 = 1008, not 1000, so the binding case is 2048-1008 = 1040.
Storing 1048 would leave H=1000 (and only H=1000) dropping 8 further steps,
reinstating a small version of the confound this exists to remove.

WHAT IS AND IS NOT PRESERVED
----------------------------
Preserved: the twelve subsets still share a byte-identical observed context,
because they shared one before and the same tail is taken from each. The
regimes, the horizons and the generative process are untouched — only the front
of the observed window is dropped.

NOT preserved: the published oracle / Chronos-2 MASE numbers in the source
corpus's README. Those were measured against 2048 steps of history and against
a seasonal-naive denominator computed from them; both change here. Numbers from
this corpus are comparable to each other, not to that table.

`start` is advanced by the number of steps dropped, so timestamps still describe
the data. Leaving it would place every series 1008 hours before its own values.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

SRC = Path("/group-volume/ts-dataset/cross_horizon_length")
DST = Path("/group-volume/ts-dataset/cross_horizon_length_trim_equal_observed")

HORIZONS = (16, 100, 400, 1000)
REGIMES = ("high", "mid", "low")

#: The observed length every subset is cut to. Derived, not chosen: it is
#: `context_length - ceil(max(H)/patch) * patch` for the checkpoint family this
#: sweep is run with (context_length 2048, patch_size 16). Recorded in the
#: corpus metadata so a run against a model with a different window can tell.
OBSERVED = 1040
REF_CONTEXT_LENGTH = 2048
REF_PATCH_SIZE = 16

#: Hours per step. The source corpus is freq "H" throughout; asserted, not
#: assumed, because advancing `start` by the wrong unit silently mis-times
#: every series.
FREQ = "H"


def effective_observed(horizon: int, context_length: int, patch: int) -> int:
    """What `append_forecast_region` leaves for the context at this horizon."""
    forecast_len = -(-horizon // patch) * patch
    return context_length - forecast_len


def trim_one(src: Path, dst: Path, horizon: int, observed: int) -> dict:
    from datasets import load_from_disk

    ds = load_from_disk(str(src))
    split = ds["train"]
    keep = observed + horizon
    n_drop = None

    def cut(batch):
        nonlocal n_drop
        targets, starts = [], []
        for t, s in zip(batch["target"], batch["start"]):
            drop = len(t) - keep
            if drop < 0:
                raise SystemExit(
                    f"{src}: series is {len(t)} steps, shorter than the "
                    f"{keep} ({observed} observed + {horizon} horizon) this "
                    f"trim needs")
            n_drop = drop if n_drop is None else n_drop
            if drop != n_drop:
                raise SystemExit(
                    f"{src}: series lengths differ ({drop} vs {n_drop} to "
                    f"drop). The corpus is supposed to be uniform; a per-series "
                    f"trim would make `start` wrong for some of them.")
            targets.append(t[drop:])
            # Advanced by exactly what was dropped: the series now BEGINS
            # `drop` steps later, and a start left at the old value would
            # mis-time every seasonal component in the data.
            starts.append(s + timedelta(hours=drop))
        return {"target": targets, "start": starts}

    split = split.map(cut, batched=True, batch_size=512,
                      desc=f"{src.parent.name}/{src.name}")
    if set(split["freq"][:1]) - {FREQ}:
        raise SystemExit(f"{src}: freq is not {FREQ!r}, so the `start` shift "
                         f"above uses the wrong unit")
    from datasets import DatasetDict
    dst.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": split}).save_to_disk(str(dst))
    return {"n_series": len(split), "steps_dropped": n_drop,
            "series_len": keep}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=SRC)
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--observed", type=int, default=OBSERVED)
    p.add_argument("--force", action="store_true",
                   help="rebuild subsets that already exist")
    args = p.parse_args()

    worst = min(effective_observed(h, REF_CONTEXT_LENGTH, REF_PATCH_SIZE)
                for h in HORIZONS)
    if args.observed > worst:
        raise SystemExit(
            f"--observed {args.observed} exceeds {worst}, the history the "
            f"longest horizon (H={max(HORIZONS)}) leaves at "
            f"context_length={REF_CONTEXT_LENGTH}, patch_size={REF_PATCH_SIZE}. "
            f"The model would truncate that subset and only that subset, which "
            f"is the confound this corpus exists to remove.")

    rows = []
    for h in HORIZONS:
        for regime in REGIMES:
            src, dst = args.src / f"h{h}" / regime, args.dst / f"h{h}" / regime
            if dst.exists() and not args.force:
                print(f"  SKIP h{h}/{regime} — exists")
                continue
            info = trim_one(src, dst, h, args.observed)
            rows.append({"horizon": h, "regime": regime, **info})
            print(f"  h{h}/{regime}: {info['n_series']} series, "
                  f"dropped {info['steps_dropped']} -> {info['series_len']} steps")

    # MERGED, not overwritten. A re-run on a host that already has the corpus
    # skips every subset, so `rows` is empty — writing it straight out would
    # replace a complete record with `"subsets": []` and destroy the only
    # on-disk statement of what was cut. That is exactly what happens when the
    # corpus is copied to the A100 and `--stage all` is used there.
    (args.dst).mkdir(parents=True, exist_ok=True)
    meta_path = args.dst / "metadata.json"
    subsets = {}
    if meta_path.is_file():
        try:
            for r in json.loads(meta_path.read_text()).get("subsets", []):
                subsets[(r["horizon"], r["regime"])] = r
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger_warn = f"could not read {meta_path} ({e}); rewriting it"
            print(f"  WARNING: {logger_warn}")
    for r in rows:
        subsets[(r["horizon"], r["regime"])] = r
    rows = [subsets[k] for k in sorted(subsets)]
    meta_path.write_text(json.dumps({
        "source": str(args.src),
        "observed_length": args.observed,
        "horizons": list(HORIZONS), "regimes": list(REGIMES),
        "cut_for": {"context_length": REF_CONTEXT_LENGTH,
                    "patch_size": REF_PATCH_SIZE},
        "why": ("every horizon shows the model the same observed history; "
                "see build_cross_horizon_trim.py"),
        "subsets": rows,
    }, indent=1, default=str))
    print(f"wrote {meta_path} ({len(rows)}/12 subsets recorded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
