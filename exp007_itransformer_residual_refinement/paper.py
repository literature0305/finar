#!/usr/bin/env python3
"""The published iTransformer numbers and the scripts that produced them.

Both halves are quoted, not guessed:

* `OFFICIAL` mirrors `scripts/multivariate_forecasting/*/iTransformer*.sh` in
  `thuml/iTransformer` — including the per-horizon `d_model` changes ETTh1
  makes and the non-default batch size / learning rate ECL and Traffic use.
  Anything those scripts leave alone is `run.py`'s default and lives in
  `DEFAULTS`.
* `PAPER` is Table 10 (ICLR 2024 camera-ready, arXiv:2310.06625v3), "Full
  results of the long-term forecasting task", iTransformer column. That table's
  caption fixes the protocol: **input sequence length 96 for all baselines**.

`precheck.py` and `build_table.py` read this; nothing trains from it implicitly.
A reproduction that lands outside `TOLERANCE` of the `PAPER` entry is reported
as a MISS, not rounded into agreement.
"""

from __future__ import annotations

#: `run.py`'s argparse defaults, for everything the scripts do not set.
DEFAULTS = dict(
    seq_len=96, n_heads=8, dropout=0.1, activation="gelu", use_norm=True,
    batch_size=32, learning_rate=1e-4, train_epochs=10, patience=3,
    lradj="type1", loss="MSE", seed=2023, freq="h", features="M",
)

#: Per dataset: `e_layers`, and `d_model`/`d_ff` either as one int or keyed by
#: `pred_len`. `batch_size` / `learning_rate` appear only where the official
#: script overrides the default.
OFFICIAL = {
    "ETTh1":    dict(e_layers=2, d_model={96: 256, 192: 256, 336: 512, 720: 512}),
    "ETTh2":    dict(e_layers=2, d_model=128),
    "ETTm1":    dict(e_layers=2, d_model=128),
    "ETTm2":    dict(e_layers=2, d_model=128),
    "ECL":      dict(e_layers=3, d_model=512, batch_size=16, learning_rate=5e-4),
    "Traffic":  dict(e_layers=4, d_model=512, batch_size=16, learning_rate=1e-3),
    "Weather":  dict(e_layers=3, d_model=512),
    "Solar":    dict(e_layers=2, d_model=512, learning_rate=5e-4),
    "Exchange": dict(e_layers=2, d_model=128),
}

#: (MSE, MAE) per dataset and prediction length, lookback 96.
PAPER = {
    "ETTm1":    {96: (0.334, 0.368), 192: (0.377, 0.391),
                 336: (0.426, 0.420), 720: (0.491, 0.459)},
    "ETTm2":    {96: (0.180, 0.264), 192: (0.250, 0.309),
                 336: (0.311, 0.348), 720: (0.412, 0.407)},
    "ETTh1":    {96: (0.386, 0.405), 192: (0.441, 0.436),
                 336: (0.487, 0.458), 720: (0.503, 0.491)},
    "ETTh2":    {96: (0.297, 0.349), 192: (0.380, 0.400),
                 336: (0.428, 0.432), 720: (0.427, 0.445)},
    "ECL":      {96: (0.148, 0.240), 192: (0.162, 0.253),
                 336: (0.178, 0.269), 720: (0.225, 0.317)},
    "Exchange": {96: (0.086, 0.206), 192: (0.177, 0.299),
                 336: (0.331, 0.417), 720: (0.847, 0.691)},
    "Traffic":  {96: (0.395, 0.268), 192: (0.417, 0.276),
                 336: (0.433, 0.283), 720: (0.467, 0.302)},
    "Weather":  {96: (0.174, 0.214), 192: (0.221, 0.254),
                 336: (0.278, 0.296), 720: (0.358, 0.347)},
    "Solar":    {96: (0.203, 0.237), 192: (0.233, 0.261),
                 336: (0.248, 0.273), 720: (0.249, 0.275)},
}

#: How far a reproduction may land from a published number before it is called
#: a miss.
#:
#: Calibrated from the paper's OWN seed noise, not from any run of this code.
#: Table 5 reports iTransformer over five seeds: the largest standard deviation
#: is 0.012 MSE on Exchange/720 and 0.006 on ECL/720, i.e. 1.4% and 2.7%
#: relative; most cells are at or below 1%. A band at 5% relative is therefore
#: under 2x the noisiest published sigma — wide enough that a different torch /
#: cuda / GPU generation does not read as a regression, narrow enough that the
#: failures this check exists for (a wrong split, a wrong scaler, a decoder
#: that never sees the covariates) are nowhere near it: those miss by tens of
#: percent, not by three.
#:
#: The absolute floor carries the small numbers, where 5% of 0.086 (Exchange/96)
#: would be tighter than the paper's own seed spread.
#:
#: For the record, the five cells reproduced locally before this value was set
#: landed at +0.2% (ETTh1/96), +1.0% (ETTh2/96), +2.7% (ETTm1/96), +3.1%
#: (ETTm2/96) and +2.2% (Weather/96) on MSE — so ETTm2/96 would be a MISS under
#: a 3% band and is a MATCH under this one. Both numbers are printed on every
#: run; the verdict is a summary of them, not a replacement.
TOLERANCE = dict(rel=0.05, abs=0.005)


def hparams(dataset: str, pred_len: int) -> dict:
    """The official settings for one cell of the table."""
    if dataset not in OFFICIAL:
        raise KeyError(f"no official script for {dataset!r} "
                       f"(have {sorted(OFFICIAL)})")
    out = dict(DEFAULTS)
    for key, value in OFFICIAL[dataset].items():
        out[key] = value[pred_len] if isinstance(value, dict) else value
    out["d_ff"] = out["d_model"]        # every official script sets them equal
    out["pred_len"] = pred_len
    return out


def target(dataset: str, pred_len: int):
    """`(mse, mae)` the paper reports, or None where it reports nothing."""
    return PAPER.get(dataset, {}).get(pred_len)


def within_tolerance(got: float, want: float) -> bool:
    return abs(got - want) <= max(TOLERANCE["abs"], TOLERANCE["rel"] * want)
