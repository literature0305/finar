#!/usr/bin/env python3
"""Train one iTransformer cell — one dataset, one prediction length.

The optimization is the official one (`experiments/exp_long_term_forecasting.py`
in `thuml/iTransformer`): Adam on MSE, validation every epoch, `type1` learning
rate halving, early stopping on validation loss with the best epoch's weights
restored at the end. Defaults come from `paper.py`, which quotes the official
scripts, so `--dataset ETTh1 --pred-len 96` alone is the paper's configuration.

When the run finishes it scores the test split itself (`run_eval.py`), so a
submitted job leaves `metrics.json` behind without a second invocation.

The refinement is off by default; `--refinement on` turns on the EO v4 chain of
encoders. The two scenarios are spelled out in `train.sh`'s usage.
"""

from __future__ import annotations

import argparse
import json
import logging
import hashlib
import random
import resource
import sys
import time
from pathlib import Path

sys.path.append(str(next(d for d in Path(__file__).resolve().parents
                         if (d / "finar_cpu.py").is_file())))
from finar_cpu import argv_workers, limit_cpu  # noqa: E402

limit_cpu(argv_workers(), quiet=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import paper  # noqa: E402
import run_eval  # noqa: E402
from run_eval import batch_to_device  # noqa: E402
from data import (DATASETS, build_loader, build_splits,  # noqa: E402
                  n_time_features)
from itransformer import ITransformer, ModelConfig, count_parameters  # noqa: E402

logger = logging.getLogger("finar_exp007")


def _bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def adjust_learning_rate(optimizer, epoch: int, lr: float, lradj: str) -> None:
    """`utils/tools.py`. `type1` halves every epoch from the second on; the
    first call (epoch 1) leaves the rate alone."""
    if lradj == "type1":
        new_lr = lr * (0.5 ** (epoch - 1))
    elif lradj == "type2":
        table = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7,
                 20: 5e-8}
        if epoch not in table:
            return
        new_lr = table[epoch]
    elif lradj == "none":
        return
    else:
        raise ValueError(f"unknown --lradj {lradj!r}")
    for group in optimizer.param_groups:
        group["lr"] = new_lr
    logger.info("  learning rate -> %g", new_lr)


class EarlyStopping:
    """`utils/tools.py`, including its tie rule: `score < best + delta` is what
    counts against the patience, so a validation loss EQUAL to the best saves
    the checkpoint and resets the counter."""

    def __init__(self, patience: int, path: Path):
        self.patience, self.path = patience, path
        self.best = None
        self.counter = 0
        self.stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> None:
        score = -val_loss
        if self.best is None or score >= self.best:
            self.best = score
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.path)
            logger.info("  val loss %.7f is the best so far — checkpoint saved",
                        val_loss)
            return
        self.counter += 1
        logger.info("  EarlyStopping counter: %d of %d", self.counter,
                    self.patience)
        if self.counter >= self.patience:
            self.stop = True


#: Written by a finished run and read back as its result. All of it has to go
#: before a rerun starts, or a crash leaves the PREVIOUS run's numbers in place
#: for `run_eval.py` and `build_table.py` to pick up as this one's.
RUN_ARTIFACTS = ("checkpoint.pt", "metrics.json", "train_log.json")

#: Arguments that change what a run IS, beyond the dataset/lengths/variant the
#: run id already spells out. Any of them at a non-default value earns the run
#: its own directory (see `_run_id`).
TUNABLES = (
    "freq", "d_model", "d_ff", "e_layers", "n_heads", "dropout", "use_norm",
    # coe_train_depth_max / residual / bottleneck are already in the variant
    # name; coe_eval_depth is not, and it changes what the run REPORTS.
    "coe_eval_depth",
    "coe_stochastic_repeat", "coe_backprop", "coe_internal_loss",
    "coe_future_marks", "batch_size", "learning_rate", "epochs", "patience",
    "lradj", "seed", "amp",
)


def _run_id(args, cfg: ModelConfig, parser_defaults: dict) -> str:
    """A directory name that is a FUNCTION of the configuration.

    `{dataset}_{seq}_{pred}_{variant}` alone does not separate two runs
    differing only in a seed, a learning rate or a backprop mode — they land in
    one directory, and then "which checkpoint is this" has no answer. Refusing
    the collision (what this used to do) turns a naming defect into an operator
    decision on every rerun; naming the runs apart removes it.

    The suffix appears only when something was overridden, so the canonical run
    of each variant keeps its readable name.
    """
    name = f"{args.dataset}_{cfg.seq_len}_{cfg.pred_len}_{cfg.variant}"
    overrides = {k: v for k, v in vars(args).items()
                 if k in TUNABLES and v != parser_defaults[k]}
    if not overrides:
        return name
    blob = json.dumps(overrides, sort_keys=True, default=str).encode()
    return f"{name}_{hashlib.sha256(blob).hexdigest()[:8]}"


def peak_resources(device: str) -> dict:
    """What this run actually needed, for sizing the machine that runs it.

    `ru_maxrss` is the high-water mark of the whole process, so it covers the
    corpus parse as well as training — which is what a host-memory limit has to
    accommodate. The CUDA figures are torch's allocator: `allocated` is the
    live tensors, `reserved` is what it took from the driver and is the number
    that has to fit in VRAM.
    """
    out = {"host_peak_gb": round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}
    if device.startswith("cuda") and torch.cuda.is_available():
        out["gpu_peak_gb"] = round(
            torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        out["gpu_reserved_gb"] = round(
            torch.cuda.max_memory_reserved() / 2 ** 30, 2)
    return out


def _claim_run_dir(run_dir: Path, config: dict) -> None:
    """Delete the previous run's results before this one starts.

    `_run_id` makes a shared directory mean a shared configuration, so there is
    nothing to refuse here — but a rerun still has to clear what is there.
    Leaving it is how a rerun that fails early gets SCORED: `config.json` is
    rewritten, the old `checkpoint.pt` survives, and `run_eval` reports the
    previous model under the new config.
    """
    for name in RUN_ARTIFACTS:
        (run_dir / name).unlink(missing_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))


#: Every argument that only means something for a chain. A baseline run
#: returns them to their defaults. `eval_depths` belongs here too: it is not a
#: `--coe-*` flag but `run_eval.resolve_depths` refuses a depth above 1 for a
#: baseline, so leaving it set killed the run AFTER training finished.
COE_ARGS = ("coe_train_depth_max", "coe_eval_depth", "coe_residual",
            "coe_bottleneck", "coe_stochastic_repeat", "coe_backprop",
            "coe_internal_loss", "coe_future_marks", "eval_depths")


def neutralise_coe(args, parser_defaults: dict) -> list[str]:
    """Return the chain settings to their defaults for a baseline run.

    A baseline has no chain to configure, and `ModelConfig` REFUSES a depth
    above 1 without one — so `--refinement off --train-depth 3` would simply
    fail. That makes the one thing this experiment is for, running the two arms
    with everything else held fixed, impossible to express as one flag set.
    Neutralising here rather than in the launcher keeps the rule in the layer
    that records it (`config.json`) and reachable by `python train.py`.

    Returns the names of the arguments it ignored, for the caller to log.
    """
    if args.refinement:
        return []
    ignored = []
    for name in COE_ARGS:
        if getattr(args, name) != parser_defaults[name]:
            ignored.append("--" + name.replace("_", "-"))
        setattr(args, name, parser_defaults[name])
    return ignored


def build_config(args) -> tuple[ModelConfig, dict]:
    """Resolve the model config: official defaults, then explicit overrides."""
    official = paper.hparams(args.dataset, args.pred_len)
    marks = DATASETS[args.dataset]["marks"]
    d_model = args.d_model or official["d_model"]
    cfg = ModelConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=d_model,
        # Every official script sets d_ff = d_model (paper.hparams pins that),
        # so --d-model alone moves both.
        d_ff=args.d_ff or d_model,
        e_layers=args.e_layers or official["e_layers"],
        n_heads=args.n_heads or official["n_heads"],
        dropout=official["dropout"] if args.dropout is None else args.dropout,
        activation=official["activation"],
        use_norm=official["use_norm"] if args.use_norm is None else args.use_norm,
        n_marks=n_time_features(args.freq) if marks else 0,
        coe_enabled=args.refinement,
        coe_train_depth_max=args.coe_train_depth_max,
        coe_eval_depth=args.coe_eval_depth,
        coe_residual=args.coe_residual,
        coe_bottleneck=args.coe_bottleneck,
        coe_stochastic_repeat=args.coe_stochastic_repeat,
        coe_backprop=args.coe_backprop,
        coe_internal_loss=args.coe_internal_loss,
        coe_future_marks=args.coe_future_marks,
    )
    optim = dict(
        batch_size=args.batch_size or official["batch_size"],
        learning_rate=(official["learning_rate"] if args.learning_rate is None
                       else args.learning_rate),
        train_epochs=args.epochs or official["train_epochs"],
        patience=args.patience or official["patience"],
        lradj=args.lradj or official["lradj"],
        seed=official["seed"] if args.seed is None else args.seed,
    )
    return cfg, optim


def _loss_for_batch(model, cfg, criterion, batch_x, batch_y, x_mark, y_mark):
    """Final-pass loss, or the mean over every pass under `coe_internal_loss`.

    Deep supervision is EO v4's: each depth gets a direct gradient instead of
    only the depth that happened to be drawn. It needs `coe_backprop='all'`
    (the config refuses otherwise), because under `'last'` the earlier passes
    are produced in `no_grad` and their loss would be a constant.
    """
    if cfg.coe_internal_loss:
        outs = model(batch_x, x_mark, y_mark, return_all_passes=True)
        return torch.stack([criterion(o, batch_y) for o in outs]).mean()
    return criterion(model(batch_x, x_mark, y_mark), batch_y)


@torch.no_grad()
def validate(model, cfg, loader, criterion, device) -> float:
    """The official `vali()`: the MEAN OF PER-BATCH MSE over a `drop_last=True`
    loader, not the global mean. Equal batch sizes make the two identical here,
    but the loader's dropped tail does not — and this number selects the
    checkpoint, so it is reproduced exactly."""
    model.eval()
    losses = []
    for batch_x, batch_y, x_mark, y_mark in loader:
        # The TARGET never goes to the device: the official vali() computes
        # this loss on the host, and moving `batch_y` there only to copy it
        # straight back doubled the transfer for an identical number.
        batch_x, _, x_mark, y_mark = batch_to_device(
            (batch_x, batch_y, x_mark, y_mark), cfg, device, targets=False)
        out = model(batch_x, x_mark, y_mark)
        losses.append(criterion(out.cpu(), batch_y.float()).item())
    model.train()
    return float(np.average(losses)) if losses else float("nan")


def train(args, parser_defaults: dict | None = None) -> dict:
    defaults = parser_defaults if parser_defaults is not None else \
        _parser_defaults()
    ignored = neutralise_coe(args, defaults)
    cfg, optim_cfg = build_config(args)
    torch.manual_seed(optim_cfg["seed"])
    np.random.seed(optim_cfg["seed"])
    random.seed(optim_cfg["seed"])

    run_id = args.run_name or _run_id(args, cfg, defaults)
    run_dir = Path(args.out) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "dataset": args.dataset, "freq": args.freq, "model": cfg.to_dict(),
        "optim": optim_cfg, "data_root": str(Path(args.data_root).resolve()),
    }
    # BEFORE the datasets and the model: clearing the previous run's results
    # after a 136 MB csv parse helps nobody.
    _claim_run_dir(run_dir, run_config)
    if ignored:
        logger.info("--refinement off: ignoring %s (a baseline has no chain "
                    "to configure)", " ".join(ignored))

    device = args.device
    # ONE read for both splits (run_eval reads the file again for the test
    # split once training finishes; that one is a separate process's worth of
    # work in the standalone case and is not worth threading through).
    splits = build_splits(args.dataset, args.data_root, ("train", "val"),
                          cfg.seq_len, cfg.pred_len, args.freq)
    loaders = {
        "train": build_loader(splits["train"], "train",
                              optim_cfg["batch_size"], args.loader_workers),
        # ZERO workers, deliberately. With persistent_workers both pools stay
        # alive once validation starts, so the process group would hold the
        # parent's own threads plus TWO worker pools — more CPU than the cap
        # allows. Validation is one pass over an in-memory array; the parent
        # does it directly and only one pool ever exists.
        "val": build_loader(splits["val"], "val", optim_cfg["batch_size"], 0),
    }
    model = ITransformer(cfg).to(device)
    logger.info("run %s — %s", run_id, cfg.variant)
    logger.info("  splits: train %d, val %d windows", len(splits["train"]),
                len(splits["val"]))
    logger.info("  %d parameters, d_model=%d d_ff=%d e_layers=%d, %d marks",
                count_parameters(model), cfg.d_model, cfg.d_ff, cfg.e_layers,
                cfg.n_marks)
    if cfg.coe_enabled:
        logger.info("  COE: K=%d eval=%d residual=%s bottleneck=%s "
                    "stochastic=%s backprop=%s internal_loss=%s",
                    cfg.coe_train_depth_max, cfg.coe_eval_depth,
                    cfg.coe_residual, cfg.coe_bottleneck,
                    cfg.coe_stochastic_repeat, cfg.coe_backprop,
                    cfg.coe_internal_loss)

    run_config["parameters"] = count_parameters(model)
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=optim_cfg["learning_rate"])
    criterion = nn.MSELoss()
    stopper = EarlyStopping(optim_cfg["patience"], run_dir / "checkpoint.pt")
    scaler = (torch.amp.GradScaler(device.split(":")[0])
              if args.amp else None)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started_run = time.time()
    history = []

    model.train()
    for epoch in range(1, optim_cfg["train_epochs"] + 1):
        started = time.time()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad()
            batch_x, batch_y, x_mark, y_mark = batch_to_device(
                batch, cfg, device)
            # `enabled=False` makes autocast a no-op, so the loss has ONE
            # call site and cannot drift between the AMP and fp32 paths.
            with torch.autocast(device_type=device.split(":")[0],
                                enabled=args.amp):
                loss = _loss_for_batch(model, cfg, criterion, batch_x,
                                       batch_y, x_mark, y_mark)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
        train_loss = float(np.average(losses))
        val_loss = validate(model, cfg, loaders["val"], criterion, device)
        logger.info("epoch %d/%d  train %.7f  val %.7f  (%.1fs)", epoch,
                    optim_cfg["train_epochs"], train_loss, val_loss,
                    time.time() - started)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss})
        stopper(val_loss, model, epoch)
        if stopper.stop:
            logger.info("early stopping at epoch %d (best was %d)", epoch,
                        stopper.best_epoch)
            break
        adjust_learning_rate(optimizer, epoch, optim_cfg["learning_rate"],
                             optim_cfg["lradj"])

    resources = peak_resources(device)
    resources["train_seconds"] = round(time.time() - started_run, 1)
    resources["epochs_run"] = len(history)
    resources["parameters"] = run_config["parameters"]
    logger.info("  peak %s", ", ".join(f"{k}={v}" for k, v in resources.items()))
    (run_dir / "train_log.json").write_text(json.dumps(
        {"best_epoch": stopper.best_epoch, "resources": resources,
         "history": history}, indent=2))
    if not (run_dir / "checkpoint.pt").is_file():
        # Every branch above saves on the first epoch, so reaching this means
        # the loop never ran a single epoch — refuse rather than evaluate a
        # randomly initialised model and report it as a result.
        raise RuntimeError(
            f"no checkpoint was written to {run_dir} — training never "
            f"completed an epoch")
    logger.info("best epoch %d; scoring the test split", stopper.best_epoch)

    del loaders, model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    # Same reason as the validation loader: the training pool is gone by now,
    # but scoring is a single pass and does not need a second one.
    return run_eval.evaluate(run_dir, args.data_root, device,
                             args.eval_batch_size, 0, args.eval_depths)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train iTransformer, with or without EO v4 refinement.")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--data-root", required=True,
                   help="root holding ETT-small/, electricity/, ... "
                        "(see prepare_data.sh)")
    p.add_argument("--out", default="./runs", help="run directories go here")
    p.add_argument("--run-name", default=None,
                   help="override the auto-generated run directory name")
    # -- task --
    p.add_argument("--seq-len", type=int, default=96,
                   help="lookback. The paper's table is 96 throughout; "
                        "changing it leaves the reproduction protocol")
    p.add_argument("--pred-len", type=int, required=True)
    p.add_argument("--freq", default="h",
                   help="timestamp-feature frequency. 'h' matches every "
                        "official script, INCLUDING the 15-minute ETTm and "
                        "10-minute weather runs (run.py's default is never "
                        "overridden there)")
    # -- model (None = the official value for this cell) --
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--e-layers", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--use-norm", type=_bool, default=None)
    # -- EO v4 chain-of-encoder --
    p.add_argument("--refinement", type=_bool, default=False,
                   help="on: apply the EO v4 chain of encoders (the "
                        "experiment). off: the paper's model (the baseline)")
    p.add_argument("--coe-train-depth-max", type=int, default=1,
                   help="K. Training draws N ~ Uniform{1..K} per step")
    p.add_argument("--coe-eval-depth", type=int, default=1,
                   help="passes at inference; MAY exceed K")
    p.add_argument("--coe-residual", type=_bool, default=True,
                   help="each pass emits a delta added to the running "
                        "forecast (default: true — this is the experiment)")
    p.add_argument("--coe-bottleneck", type=_bool, default=True,
                   help="true: feed the forecast back through the embedding. "
                        "false: EO v4's hidden-state chain ablation")
    p.add_argument("--coe-stochastic-repeat", type=_bool, default=True)
    p.add_argument("--coe-backprop", choices=["last", "all"], default="last")
    p.add_argument("--coe-internal-loss", type=_bool, default=False,
                   help="supervise every pass, not just the last "
                        "(requires --coe-backprop all)")
    p.add_argument("--coe-future-marks", type=_bool, default=True,
                   help="give the fed-back window the horizon's real "
                        "timestamps, as EO v4's time channel does")
    # -- optimization (None = the official value) --
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--lradj", choices=["type1", "type2", "none"], default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--amp", action="store_true",
                   help="mixed precision. OFF by default: the published "
                        "numbers are fp32 and the official scripts pass no "
                        "--use_amp")
    # -- runtime --
    p.add_argument("--num-workers", type=int, default=None,
                   help="CPU cap for this run. Default: derived from the "
                        "scheduler affinity mask and the cgroup quota. A value "
                        "above the allocation is clamped, not obeyed.")
    p.add_argument("--loader-workers", type=int, default=0,
                   help="DataLoader worker PROCESSES. Default 0 — the "
                        "dataset is an in-memory numpy array, so a worker "
                        "adds IPC and nothing else: measured 1.5-2.0 s/epoch "
                        "at 0 against 2.8-3.4 s at 2 on ETTh1, and 9 s "
                        "against 13 s on Weather. Workers are also reserved "
                        "out of the parent's thread pool, because the CPU cap "
                        "is a budget for the process group.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--eval-depths", nargs="+", default=None,
                   help="score these chain depths when training finishes")
    return p.parse_args(argv)


def _parser_defaults() -> dict:
    """What every optional argument defaults to, taken from the parser itself.

    `_run_id` and `neutralise_coe` both ask "was this overridden", and this is
    the one place that answers it — a hand-kept second copy of the defaults is
    exactly what would drift. The three required arguments are given dummy
    values; none of them is in `TUNABLES`.
    """
    return vars(parse_args(["--dataset", sorted(DATASETS)[0],
                            "--pred-len", "96", "--data-root", "."]))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        datefmt="%m-%d %H:%M:%S")
    defaults = _parser_defaults()
    cap = limit_cpu(args.num_workers)
    # The cap is a budget for the PROCESS GROUP, not per pool. Loader workers
    # are processes of their own, so they are reserved out of the parent's
    # thread pool rather than added on top of it.
    args.loader_workers = max(0, min(args.loader_workers, cap - 1))
    if args.loader_workers:
        limit_cpu(cap - args.loader_workers)
    for name in ("epochs", "batch_size", "patience"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1 "
                             f"(got {value})")
    train(args, defaults)


if __name__ == "__main__":
    main()
