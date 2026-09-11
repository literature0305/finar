#!/usr/bin/env python3
"""The LTSF benchmark loaders, as `thuml/iTransformer` defines them.

The splits, the scaler and the timestamp features decide the numbers as much as
the model does, so these are transcribed from the official
`data_provider/data_loader.py` rather than re-derived: ETT's calendar-month
borders (12/4/4 months, NOT a ratio), the 0.7/0.1/0.2 ratio split for the
`custom` csvs, Solar's 0.7/0.1/0.2 on its own file, a `StandardScaler` fitted on
the TRAIN slice only, and each split starting `seq_len` steps early so its first
window has a full lookback.

`precheck.py` compares every split length AND the first/last window of each
against the official loader, so a drift here is caught before a run, not after.

Two deliberate simplifications, both verified to change nothing for this model:

* `label_len` is dropped (the official value is 48). iTransformer has no
  decoder; the official code slices `batch_y[:, -pred_len:, :]` everywhere, so
  the label prefix is loaded and discarded. `seq_y` here is the horizon only.
* The test loader may use a batch size > 1. The official one is `batch_size=1,
  drop_last=True`, which drops nothing; MSE/MAE are computed over the
  concatenated predictions, so any batch size with `drop_last=False` yields the
  same number. VALIDATION keeps `drop_last=True` and per-batch averaging,
  because that one does drop a tail and does drive early stopping.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# timestamp features (`utils/timefeatures.py`, timeenc=1 path)
# ---------------------------------------------------------------------------
def _hour_of_day(i): return i.hour / 23.0 - 0.5
def _day_of_week(i): return i.dayofweek / 6.0 - 0.5
def _day_of_month(i): return (i.day - 1) / 30.0 - 0.5
def _day_of_year(i): return (i.dayofyear - 1) / 365.0 - 0.5
def _minute_of_hour(i): return i.minute / 59.0 - 0.5
def _month_of_year(i): return (i.month - 1) / 11.0 - 0.5


#: Keyed by the pandas offset alias the official `to_offset` lookup resolves to.
#: Only the frequencies this benchmark uses are listed; `run.py`'s default is
#: 'h' and NONE of the official scripts override it, so even the 15-minute ETTm
#: and 10-minute weather runs take the 4-feature hourly set. Reproducing that is
#: the point — deriving a "better" feature set from the sampling rate would
#: change the published numbers.
_FEATURES = {
    "h": [_hour_of_day, _day_of_week, _day_of_month, _day_of_year],
    "t": [_minute_of_hour, _hour_of_day, _day_of_week, _day_of_month,
          _day_of_year],
    "d": [_day_of_week, _day_of_month, _day_of_year],
    "m": [_month_of_year],
}


def time_features(dates: pd.DatetimeIndex, freq: str = "h") -> np.ndarray:
    key = freq.lower().lstrip("0123456789")
    key = {"min": "t", "s": "t", "b": "d", "w": "d"}.get(key, key)
    if key not in _FEATURES:
        raise ValueError(f"unsupported freq {freq!r} (have {sorted(_FEATURES)})")
    return np.vstack([f(dates) for f in _FEATURES[key]]).transpose(1, 0)


def n_time_features(freq: str = "h") -> int:
    return time_features(pd.DatetimeIndex(["2020-01-01"]), freq).shape[1]


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
class _Windows(Dataset):
    """One split's sliding windows over an already-scaled corpus.

    The corpus is read and scaled ONCE per dataset (`_read_*` below) and the
    three splits are views into it. The official loader re-reads and re-scales
    the whole file for every split; on Traffic that is three parses of a 136 MB
    csv per run, and the three results are identical outside their slice.
    """

    def __init__(self, data, stamp, border1, border2, seq_len, pred_len):
        self.seq_len, self.pred_len = seq_len, pred_len
        self.data = data[border1:border2]
        self.stamp = None if stamp is None else stamp[border1:border2]

    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len
        seq_x = self.data[index:s_end]
        seq_y = self.data[s_end:r_end]
        if self.stamp is None:
            empty = np.zeros((0,), dtype=np.float32)
            return seq_x, seq_y, empty, empty
        return seq_x, seq_y, self.stamp[index:s_end], self.stamp[s_end:r_end]

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1


def _scale(values: np.ndarray, train_slice: np.ndarray) -> np.ndarray:
    """`sklearn.StandardScaler` fitted on the train slice, reimplemented so
    this module has no sklearn dependency. `ddof=0`, as sklearn uses, and the
    same zero-variance guard (a constant column is left untouched)."""
    mean = train_slice.mean(axis=0)
    std = train_slice.std(axis=0)
    return (values - mean) / np.where(std == 0.0, 1.0, std)


def _read_ett(root, file, seq_len, freq, unit):
    """ETTh*/ETTm* — calendar borders, `unit` steps per hour."""
    df_raw = pd.read_csv(os.path.join(root, file))
    m = 30 * 24 * unit
    border1s = [0, 12 * m - seq_len, 16 * m - seq_len]
    border2s = [12 * m, 16 * m, 20 * m]
    values = df_raw[df_raw.columns[1:]].to_numpy(dtype=np.float64)
    data = _scale(values, values[border1s[0]:border2s[0]])
    stamp = time_features(pd.to_datetime(df_raw["date"].values), freq)
    return (data.astype(np.float32), stamp.astype(np.float32),
            border1s, border2s)


def _read_custom(root, file, seq_len, freq, target="OT"):
    """ECL / Traffic / Weather / Exchange — 0.7 / 0.1 / 0.2 by row count."""
    df_raw = pd.read_csv(os.path.join(root, file))
    # The official loader MOVES `target` to the last column; the column order
    # decides which variate is which, so it is reproduced even though every run
    # here is `features=M` and scores all of them.
    cols = [c for c in df_raw.columns if c not in (target, "date")]
    n = len(df_raw)
    num_train, num_test = int(n * 0.7), int(n * 0.2)
    num_vali = n - num_train - num_test
    border1s = [0, num_train - seq_len, n - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, n]
    values = df_raw[cols + [target]].to_numpy(dtype=np.float64)
    data = _scale(values, values[border1s[0]:border2s[0]])
    stamp = time_features(pd.to_datetime(df_raw["date"].values), freq)
    return (data.astype(np.float32), stamp.astype(np.float32),
            border1s, border2s)


def _read_solar(root, file, seq_len, freq=None):
    """Solar-Energy — a headerless csv of floats, no timestamps at all.

    Note the official split: 0.7 train / 0.1 val / 0.2 test, where val is
    `int(0.1 * n)` rather than the remainder, so a few rows between val and
    test belong to no split. Reproduced deliberately.
    """
    values = np.loadtxt(os.path.join(root, file), delimiter=",")
    n = len(values)
    num_train, num_test, num_valid = int(n * 0.7), int(n * 0.2), int(n * 0.1)
    border1s = [0, num_train - seq_len, n - num_test - seq_len]
    border2s = [num_train, num_train + num_valid, n]
    data = _scale(values, values[border1s[0]:border2s[0]])
    return data.astype(np.float32), None, border1s, border2s


#: name -> how to read it, and what a correct copy of it looks like.
#:
#: `marks` is False where the official experiment code forces
#: `batch_x_mark = None` (Solar and PEMS). `enc_in` and `rows` are the paper's
#: shape (Table 4 of the paper / the files the official scripts run on) and are
#: ASSERTED against what is on disk, never used in its place. `sha256` pins the
#: copy this experiment was prepared from; it is ADVISORY, because the same
#: series legitimately round-trips to different bytes through a different
#: mirror, while a different shape never does.
DATASETS = {
    "ETTh1":   dict(read=_read_ett, root="ETT-small", file="ETTh1.csv",
                    unit=1, enc_in=7, rows=17420, marks=True,
                    sha256="f18de3ad269cef59"),
    "ETTh2":   dict(read=_read_ett, root="ETT-small", file="ETTh2.csv",
                    unit=1, enc_in=7, rows=17420, marks=True,
                    sha256="a3dc2c597b9218c7"),
    "ETTm1":   dict(read=_read_ett, root="ETT-small", file="ETTm1.csv",
                    unit=4, enc_in=7, rows=69680, marks=True,
                    sha256="6ce1759b1a18e332"),
    "ETTm2":   dict(read=_read_ett, root="ETT-small", file="ETTm2.csv",
                    unit=4, enc_in=7, rows=69680, marks=True,
                    sha256="db973ca252c6410a"),
    "ECL":     dict(read=_read_custom, root="electricity",
                    file="electricity.csv", enc_in=321, rows=26304,
                    marks=True, sha256="7e45845d54c5219b"),
    "Traffic": dict(read=_read_custom, root="traffic", file="traffic.csv",
                    enc_in=862, rows=17544, marks=True,
                    sha256="cb06463d56fa17d8"),
    "Weather": dict(read=_read_custom, root="weather", file="weather.csv",
                    enc_in=21, rows=52696, marks=True,
                    sha256="34ee981d07313e51"),
    "Exchange": dict(read=_read_custom, root="exchange_rate",
                     file="exchange_rate.csv", enc_in=8, rows=7588,
                     marks=True, sha256="48b4d9d3d508f510"),
    "Solar":   dict(read=_read_solar, root="Solar", file="solar_AL.txt",
                    enc_in=137, rows=52560, marks=False,
                    sha256="230327ef72d2abb3"),
}


def dataset_path(data_root: str, name: str) -> str:
    spec = DATASETS[name]
    return os.path.join(data_root, spec["root"], spec["file"])


#: Index of each split in the `border1s` / `border2s` lists every `_read_*`
#: returns, in the official loader's order.
_SPLIT = {"train": 0, "val": 1, "test": 2}


def build_splits(name: str, data_root: str, flags, seq_len: int,
                 pred_len: int, freq: str = "h") -> dict:
    """Several splits of one dataset from ONE read, with the shape checked.

    A silently wrong variate count is the failure the check exists for: every
    reader here takes "all columns but `date`", so a corrupted download or a
    re-exported csv would train a differently shaped model on a differently
    shaped dataset and still produce an MSE to compare with the paper's.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r} (have {sorted(DATASETS)})")
    spec = DATASETS[name]
    root = os.path.join(data_root, spec["root"])
    kw = {"unit": spec["unit"]} if "unit" in spec else {}
    data, stamp, border1s, border2s = spec["read"](
        root, spec["file"], seq_len, freq, **kw)
    got = data.shape[1]
    if got != spec["enc_in"]:
        raise ValueError(
            f"{name}: the file at {os.path.join(root, spec['file'])} has {got} "
            f"variates, the paper's benchmark has {spec['enc_in']} — re-run "
            f"prepare_data.sh, this is not the same dataset.")
    out = {}
    for flag in flags:
        i = _SPLIT[flag]
        ds = _Windows(data, stamp, border1s[i], border2s[i], seq_len, pred_len)
        if len(ds) <= 0:
            raise ValueError(
                f"{name}/{flag}: seq_len={seq_len} + pred_len={pred_len} does "
                f"not fit in the split ({len(ds.data)} rows).")
        out[flag] = ds
    return out


def build_dataset(name: str, data_root: str, flag: str, seq_len: int,
                  pred_len: int, freq: str = "h"):
    """One split. Prefer `build_splits` when more than one is wanted."""
    return build_splits(name, data_root, [flag], seq_len, pred_len, freq)[flag]


def _worker_init(_worker_id):
    # Each loader worker inherits OMP_NUM_THREADS from the parent, so N workers
    # would otherwise be entitled to N x the cap. Windowing is numpy slicing;
    # one thread each is all it needs.
    torch.set_num_threads(1)


def build_loader(dataset, flag: str, batch_size: int, workers: int):
    """Train shuffles and drops the tail; val drops the tail unshuffled (that
    is what the official vali loss averages over); test keeps every window."""
    train = flag == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        drop_last=flag in ("train", "val"),
        worker_init_fn=_worker_init if workers > 0 else None,
        persistent_workers=workers > 0,
    )
