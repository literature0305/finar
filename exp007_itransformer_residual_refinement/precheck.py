#!/usr/bin/env python3
"""Before training: is this actually iTransformer, and will it reproduce?

A reimplementation that is subtly not the paper's model still trains, still
converges, and still produces an MSE — it just answers a different question.
The same is true of a refinement whose feedback never reaches the model: it
produces a plausible "no improvement" that is indistinguishable from a real
null. So `train.sh` runs this first, and it REFUSES rather than reporting a
comfortable pass when it could not actually check.

WHAT IT CHECKS
--------------
Against a `thuml/iTransformer` checkout (`--reference`, cloned by
`prepare_data.sh --with-reference`):

  hparams   `paper.py`'s per-cell settings are what the official
            `scripts/multivariate_forecasting/*` shell scripts pass.
  model     the official `Model` and this `ITransformer` accept the same
            `state_dict` and return bit-identical forecasts.
  data      every split's window count, and its first/middle/last window,
            match the official loader for each dataset present on disk.

Without a reference (only under an explicit `--no-reference`):

  refine    R1 a single pass is identical with and without `coe_residual`
            (EO v4's invariant: A_0 = 0).
            R2 with the feedback slot's weights zeroed and no residual, the
            chain at ANY depth equals the baseline model exactly — the widened
            embedding is a pure superset, not a different architecture.
            R3 with real weights, depth 2 differs from depth 1 and depth 3
            from depth 2 — the forecast that is written back actually reaches
            the model. This is the check that separates "the refinement did
            not help" from "the refinement never ran".
            R4 the same for the hidden-state chain.
  table     the paper's targets are present for every dataset x horizon.

`--reproduce DATASET/PRED_LEN` additionally TRAINS that cell with the official
settings and compares the result to Table 10. That is the only check that
answers the reproduction question outright; the rest make it cheap to trust.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import shlex
import sys
from pathlib import Path

sys.path.append(str(next(d for d in Path(__file__).resolve().parents
                         if (d / "finar_cpu.py").is_file())))
from finar_cpu import argv_workers, limit_cpu  # noqa: E402

limit_cpu(argv_workers(), quiet=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import paper  # noqa: E402
from data import DATASETS, build_dataset, dataset_path, n_time_features  # noqa: E402
from itransformer import ITransformer, ModelConfig, count_parameters  # noqa: E402

logger = logging.getLogger("finar_exp007")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclasses.dataclass
class Result:
    name: str
    status: str
    detail: str


# ---------------------------------------------------------------------------
# the official checkout
# ---------------------------------------------------------------------------
def _import_reference(reference: Path):
    """Import the official `Model` and loaders from a checkout.

    `layers/SelfAttention_Family.py` imports `reformer_pytorch` at module
    scope for iReformer's LSH attention, which iTransformer never constructs.
    A stub keeps the precheck from requiring a dependency of a model that is
    not under test; if the real package is installed it is used instead.
    """
    import types

    if "reformer_pytorch" not in sys.modules:
        try:
            import reformer_pytorch  # noqa: F401
        except ImportError:
            stub = types.ModuleType("reformer_pytorch")
            stub.LSHSelfAttention = object
            sys.modules["reformer_pytorch"] = stub
    sys.path.insert(0, str(reference))
    try:
        from data_provider import data_loader as ref_data
        from model.iTransformer import Model as RefModel
    finally:
        sys.path.remove(str(reference))
    return RefModel, ref_data


def _ref_configs(cfg: ModelConfig):
    """The argparse namespace the official `Model.__init__` reads."""
    return argparse.Namespace(
        seq_len=cfg.seq_len, pred_len=cfg.pred_len, output_attention=False,
        use_norm=cfg.use_norm, d_model=cfg.d_model, embed="timeF", freq="h",
        dropout=cfg.dropout, class_strategy="projection", factor=1,
        n_heads=cfg.n_heads, e_layers=cfg.e_layers, d_ff=cfg.d_ff,
        activation=cfg.activation,
    )


# ---------------------------------------------------------------------------
# checks that need the reference
# ---------------------------------------------------------------------------
#: `--data_path` is what identifies a dataset unambiguously across the official
#: scripts; `--model_id` varies in case and separator.
_BY_DATA_PATH = {
    "ETTh1.csv": "ETTh1", "ETTh2.csv": "ETTh2", "ETTm1.csv": "ETTm1",
    "ETTm2.csv": "ETTm2", "electricity.csv": "ECL", "traffic.csv": "Traffic",
    "weather.csv": "Weather", "solar_AL.txt": "Solar",
    "exchange_rate.csv": "Exchange",
}


def _parse_official_scripts(reference: Path) -> dict:
    """`{(dataset, pred_len): {flag: value}}` from the official shell scripts."""
    root = reference / "scripts" / "multivariate_forecasting"
    runs = {}
    for script in sorted(root.rglob("iTransformer*.sh")):
        text = script.read_text()
        # Line continuations first: each `python -u run.py ...` invocation is
        # one logical command spread over ~20 backslash-continued lines.
        text = text.replace("\\\n", " ")
        for block in text.split("python -u run.py")[1:]:
            block = block.split("\n")[0]
            tokens = shlex.split(block)
            flags = {}
            i = 0
            while i < len(tokens):
                if tokens[i].startswith("--"):
                    key = tokens[i][2:]
                    if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                        flags[key] = tokens[i + 1]
                        i += 1
                    else:
                        flags[key] = "1"
                i += 1
            name = _BY_DATA_PATH.get(flags.get("data_path", ""))
            if name and "pred_len" in flags:
                runs[(name, int(flags["pred_len"]))] = flags
    return runs


def check_hparams(reference: Path) -> Result:
    runs = _parse_official_scripts(reference)
    if not runs:
        return Result("hparams", FAIL,
                      f"no `python -u run.py` invocations found under "
                      f"{reference}/scripts/multivariate_forecasting")
    checked, bad = 0, []
    for (name, pred_len), flags in sorted(runs.items()):
        if name not in paper.OFFICIAL:
            continue
        want = paper.hparams(name, pred_len)
        for key in ("seq_len", "e_layers", "d_model", "d_ff", "batch_size",
                    "learning_rate"):
            # Absent in the script means run.py's default, which is what
            # paper.DEFAULTS holds.
            got = float(flags[key]) if key in flags else float(want[key])
            if abs(got - float(want[key])) > 1e-12:
                bad.append(f"{name}/{pred_len} {key}: script {got:g}, "
                           f"paper.py {float(want[key]):g}")
        checked += 1
    if bad:
        return Result("hparams", FAIL, "; ".join(bad[:6]))
    return Result("hparams", PASS,
                  f"{checked} official run(s) agree with paper.py")


def check_model(reference: Path, device: str) -> Result:
    RefModel, _ = _import_reference(reference)
    worst = 0.0
    for name, pred_len in (("ETTh1", 96), ("Weather", 336)):
        hp = paper.hparams(name, pred_len)
        cfg = ModelConfig(
            seq_len=hp["seq_len"], pred_len=pred_len, d_model=hp["d_model"],
            d_ff=hp["d_ff"], e_layers=hp["e_layers"], n_heads=hp["n_heads"],
            dropout=hp["dropout"], activation=hp["activation"],
            use_norm=hp["use_norm"], n_marks=n_time_features("h"))
        torch.manual_seed(0)
        ref = RefModel(_ref_configs(cfg)).float().to(device).eval()
        ours = ITransformer(cfg).to(device).eval()
        # strict=True: a missing or extra parameter is a structural difference,
        # which is exactly what this check is for.
        ours.load_state_dict(ref.state_dict(), strict=True)
        if count_parameters(ref) != count_parameters(ours):
            return Result("model", FAIL,
                          f"{name}/{pred_len}: {count_parameters(ref)} "
                          f"parameters vs {count_parameters(ours)}")
        torch.manual_seed(1)
        n_var = DATASETS[name]["enc_in"]
        x = torch.randn(4, cfg.seq_len, n_var, device=device)
        xm = torch.randn(4, cfg.seq_len, cfg.n_marks, device=device)
        with torch.no_grad():
            want = ref(x, xm, None, None)
            got = ours(x, xm)
        if want.shape != got.shape:
            return Result("model", FAIL,
                          f"{name}/{pred_len}: shape {tuple(want.shape)} vs "
                          f"{tuple(got.shape)}")
        worst = max(worst, float((want - got).abs().max()))
    if worst != 0.0:
        return Result("model", FAIL,
                      f"forecasts differ from the official model by up to "
                      f"{worst:.3e} — this is not the same computation")
    return Result("model", PASS,
                  "identical state_dict and bit-identical forecasts on "
                  "ETTh1/96 and Weather/336")


def check_data(reference: Path, data_root: str, seq_len: int,
               pred_len: int) -> Result:
    _, ref_data = _import_reference(reference)
    ref_cls = {
        "ETTh1": ref_data.Dataset_ETT_hour, "ETTh2": ref_data.Dataset_ETT_hour,
        "ETTm1": ref_data.Dataset_ETT_minute,
        "ETTm2": ref_data.Dataset_ETT_minute,
        "ECL": ref_data.Dataset_Custom, "Traffic": ref_data.Dataset_Custom,
        "Weather": ref_data.Dataset_Custom,
        "Exchange": ref_data.Dataset_Custom, "Solar": ref_data.Dataset_Solar,
    }
    present = [n for n in sorted(DATASETS)
               if Path(dataset_path(data_root, n)).is_file()]
    if not present:
        return Result("data", SKIP,
                      f"no dataset files under {data_root} — run "
                      f"prepare_data.sh first")
    notes, bad = [], []
    for name in present:
        spec = DATASETS[name]
        root = str(Path(data_root) / spec["root"])
        for flag in ("train", "val", "test"):
            mine = build_dataset(name, data_root, flag, seq_len, pred_len)
            theirs = ref_cls[name](
                root_path=root, flag=flag, size=[seq_len, 48, pred_len],
                features="M", data_path=spec["file"], target="OT",
                timeenc=1, freq="h")
            if len(mine) != len(theirs):
                bad.append(f"{name}/{flag}: {len(mine)} windows vs "
                           f"{len(theirs)}")
                continue
            for idx in {0, len(mine) // 2, len(mine) - 1}:
                mx, my, mxm, mym = mine[idx]
                tx, ty, txm, tym = theirs[idx]
                pairs = [("x", mx, tx), ("y", my, np.asarray(ty)[-pred_len:])]
                if spec["marks"]:
                    pairs += [("x_mark", mxm, txm),
                              ("y_mark", mym, np.asarray(tym)[-pred_len:])]
                for what, a, b in pairs:
                    # float32 on BOTH sides, and then exact. The official
                    # loader hands back float64 and the training loop casts it
                    # with `.float()`, so float32 is what actually reaches the
                    # model; comparing this side's float32 against their
                    # float64 would report the cast itself as a difference
                    # (measured: 1.8e-6 on Traffic, whose z-scores reach 39).
                    a = np.asarray(a, np.float32)
                    b = np.asarray(b, np.float32)
                    if a.shape != b.shape:
                        bad.append(f"{name}/{flag}[{idx}].{what} shape "
                                   f"{a.shape} vs {b.shape}")
                        break
                    if not np.array_equal(a, b, equal_nan=True):
                        bad.append(f"{name}/{flag}[{idx}].{what} differs "
                                   f"(max {np.abs(a.astype(np.float64) - b).max():.3e})")
                        break
        notes.append(f"{name}({len(mine)} test windows)")
    if bad:
        return Result("data", FAIL, "; ".join(bad[:6]))
    return Result("data", PASS, "splits and windows match: " + ", ".join(notes))


# ---------------------------------------------------------------------------
# checks that stand alone
# ---------------------------------------------------------------------------
def _refine_cfg(**over) -> ModelConfig:
    base = dict(seq_len=96, pred_len=96, d_model=64, d_ff=64, e_layers=2,
                n_heads=4, dropout=0.0, n_marks=4, coe_enabled=True,
                coe_train_depth_max=3, coe_eval_depth=3)
    base.update(over)
    return ModelConfig(**base)


def check_refinement(device: str) -> Result:
    torch.manual_seed(7)
    n_var, notes = 7, []
    x = torch.randn(3, 96, n_var, device=device)
    xm = torch.randn(3, 96, 4, device=device)
    ym = torch.randn(3, 96, 4, device=device)

    # R1 — A_0 = 0, so one pass is the same with and without the residual.
    torch.manual_seed(11)
    res = ITransformer(_refine_cfg(coe_residual=True)).to(device).eval()
    nores = ITransformer(_refine_cfg(coe_residual=False)).to(device).eval()
    nores.load_state_dict(res.state_dict())
    with torch.no_grad():
        d1 = float((res(x, xm, ym, depth=1)
                    - nores(x, xm, ym, depth=1)).abs().max())
    if d1 != 0.0:
        return Result("refine", FAIL,
                      f"R1: depth 1 differs by {d1:.3e} between "
                      f"coe_residual true/false; A_0 is not zero")
    notes.append("R1 depth-1 residual invariance")

    # R2 — zero the feedback slot: the chain must collapse onto the baseline
    # at every depth, proving the widened embedding adds nothing else.
    torch.manual_seed(13)
    cfg = _refine_cfg(coe_residual=False, coe_future_marks=False)
    chain = ITransformer(cfg).to(device).eval()
    with torch.no_grad():
        chain.enc_embedding.value_embedding.weight[:, cfg.seq_len:] = 0.0
    base_cfg = ModelConfig(**{**dataclasses.asdict(cfg), "coe_enabled": False,
                              "coe_train_depth_max": 1, "coe_eval_depth": 1})
    base = ITransformer(base_cfg).to(device).eval()
    state = dict(chain.state_dict())
    key = "enc_embedding.value_embedding.weight"
    state[key] = state[key][:, :cfg.seq_len].contiguous()
    base.load_state_dict(state, strict=True)
    with torch.no_grad():
        want = base(x, xm)
        worst = max(float((chain(x, xm, ym, depth=d) - want).abs().max())
                    for d in (1, 2, 5))
    if worst != 0.0:
        return Result("refine", FAIL,
                      f"R2: with the feedback slot zeroed the chain still "
                      f"moves by {worst:.3e} from the baseline — the widened "
                      f"embedding changes more than the feedback")
    notes.append("R2 feedback slot is the only addition")

    # R3 — and with real weights it MUST move. A refinement that silently does
    # nothing reads exactly like a refinement that does not help.
    torch.manual_seed(17)
    model = ITransformer(_refine_cfg()).to(device).eval()
    with torch.no_grad():
        outs = model(x, xm, ym, depth=3, return_all_passes=True)
    steps = [float((outs[i + 1] - outs[i]).abs().mean()) for i in range(2)]
    if min(steps) <= 0.0:
        return Result("refine", FAIL,
                      f"R3: the chain does not move (mean |delta| per pass "
                      f"{steps}) — the fed-back forecast never reaches the "
                      f"model")
    notes.append(f"R3 chain moves (mean |delta| {steps[0]:.3e}, {steps[1]:.3e})")

    # R4 — the hidden-state ablation must also advance with depth.
    torch.manual_seed(19)
    hidden = ITransformer(
        _refine_cfg(coe_bottleneck=False)).to(device).eval()
    with torch.no_grad():
        houts = hidden(x, xm, ym, depth=3, return_all_passes=True)
    hsteps = [float((houts[i + 1] - houts[i]).abs().mean()) for i in range(2)]
    if min(hsteps) <= 0.0:
        return Result("refine", FAIL,
                      f"R4: the hidden-state chain does not move {hsteps}")
    notes.append(f"R4 hidden chain moves ({hsteps[0]:.3e}, {hsteps[1]:.3e})")
    return Result("refine", PASS, "; ".join(notes))


def check_table() -> Result:
    missing = [f"{name}/{h}" for name in paper.OFFICIAL
               for h in (96, 192, 336, 720)
               if paper.target(name, h) is None]
    if missing:
        return Result("table", FAIL,
                      f"no published target for {', '.join(missing)}")
    cells = sum(len(v) for v in paper.PAPER.values())
    return Result("table", PASS,
                  f"{cells} published (MSE, MAE) targets, lookback 96, "
                  f"tolerance {paper.TOLERANCE['rel']:.0%} / "
                  f"{paper.TOLERANCE['abs']}")


# ---------------------------------------------------------------------------
# the real thing
# ---------------------------------------------------------------------------
def check_reproduce(spec: str, data_root: str, device: str, out: str,
                    workers: int) -> Result:
    """Train one cell with the official settings and compare to Table 10."""
    import train as train_mod

    name, _, horizon = spec.partition("/")
    if name not in paper.OFFICIAL or not horizon.isdigit():
        return Result("reproduce", FAIL,
                      f"--reproduce takes DATASET/PRED_LEN, e.g. ETTh1/96 "
                      f"(got {spec!r})")
    pred_len = int(horizon)
    args = train_mod.parse_args([
        "--dataset", name, "--pred-len", str(pred_len),
        "--data-root", data_root, "--out", out, "--device", device,
        "--run-name", f"precheck_{name}_{pred_len}",
        "--loader-workers", str(workers),
    ])
    result = train_mod.train(args)
    want = paper.target(name, pred_len)
    detail = (f"{name}/{pred_len}: MSE {result['mse']:.4f} vs {want[0]:.3f}, "
              f"MAE {result['mae']:.4f} vs {want[1]:.3f}")
    ok = (paper.within_tolerance(result["mse"], want[0])
          and paper.within_tolerance(result["mae"], want[1]))
    return Result("reproduce", PASS if ok else FAIL, detail)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pre-training checks for exp007.")
    p.add_argument("--reference", default=None,
                   help="a thuml/iTransformer checkout. Without it the model "
                        "and data checks cannot run and this exits non-zero "
                        "unless --no-reference is given.")
    p.add_argument("--no-reference", action="store_true",
                   help="run only the standalone checks, and say so loudly")
    p.add_argument("--data-root", default=None,
                   help="root holding ETT-small/, electricity/, ...")
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=96)
    p.add_argument("--reproduce", default=None, metavar="DATASET/PRED_LEN",
                   help="also TRAIN that cell and compare to the paper")
    p.add_argument("--out", default="./runs")
    p.add_argument("--num-workers", type=int, default=None,
                   help="CPU cap. Default: derived from the allocation.")
    p.add_argument("--loader-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        datefmt="%m-%d %H:%M:%S")
    cap = limit_cpu(args.num_workers)

    results = [check_table(), check_refinement(args.device)]
    reference = Path(args.reference).resolve() if args.reference else None
    if reference and not (reference / "model" / "iTransformer.py").is_file():
        results.append(Result(
            "reference", FAIL,
            f"{reference} is not a thuml/iTransformer checkout "
            f"(no model/iTransformer.py)"))
        reference = None
    if reference:
        results.append(check_hparams(reference))
        results.append(check_model(reference, args.device))
        if args.data_root:
            results.append(check_data(reference, args.data_root,
                                      args.seq_len, args.pred_len))
        else:
            results.append(Result("data", SKIP, "no --data-root given"))
    else:
        why = ("--no-reference: the official checkout was not consulted"
               if args.no_reference else
               "no --reference given — pass one, or --no-reference to accept "
               "an unverified model")
        for name in ("hparams", "model", "data"):
            results.append(Result(name, SKIP, why))
    if args.reproduce:
        results.append(check_reproduce(
            args.reproduce, args.data_root, args.device, args.out,
            min(args.loader_workers, cap)))

    width = max(len(r.name) for r in results)
    print("\n" + "=" * 72)
    print("  exp007 precheck")
    print("=" * 72)
    for r in results:
        print(f"  [{r.status}] {r.name:<{width}}  {r.detail}")
    print("=" * 72)

    failed = [r.name for r in results if r.status == FAIL]
    skipped = [r.name for r in results if r.status == SKIP]
    if failed:
        print(f"  FAILED: {', '.join(failed)}\n")
        sys.exit(1)
    if skipped and not args.no_reference:
        print(f"  INCOMPLETE: {', '.join(skipped)} could not run. Pass "
              f"--reference (prepare_data.sh --with-reference) or accept the "
              f"gap explicitly with --no-reference.\n")
        sys.exit(2)
    if skipped:
        print(f"  PASSED, WITH {len(skipped)} CHECK(S) NOT RUN: "
              f"{', '.join(skipped)}\n")
    else:
        print("  All checks passed.\n")


if __name__ == "__main__":
    main()
