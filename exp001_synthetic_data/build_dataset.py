#!/usr/bin/env python3
"""Build the FiNAR exp001 corpora: one length, two cross-variate axes.

exp001 asks whether iterative refinement buys more when the forecast target
carries more dependency. The previous build answered the cross-horizon half and
failed the cross-variate half, for reasons that are recorded here because they
are what this file is shaped to avoid.

=============================================================================
WHAT WENT WRONG BEFORE, AND WHAT CHANGED
=============================================================================
1. THE CROSS-VARIATE KNOB NEVER REACHED THE DATA. The old build added an
   independent per-variate seasonal backbone `m` on top of the structured
   part, and `m` carried 86.7% of the observed variance. A nominal rho of 0.9
   therefore appeared in the saved series as a correlation of 0.09, and the
   diagnostics did not catch it because they measured the correlation of the
   LATENT component, not of the series that was written. Measured on the
   shipped corpus, rho=0.9 and rho=0.0 differ by 0.008 in mean |corr|.

   -> There is no backbone. The generator's output IS the observed series, so
      the structured part carries the whole variance, and `diagnostics.json`
      records the correlation of the SAVED SERIES (`observed_*`) alongside the
      latent one.

2. THE HORIZON AXIS WAS CONFOUNDED WITH THE CORPUS. A separate corpus was
   generated per horizon (seeded `seed + H`), so reading a trend down the H
   axis compared different draws. -> ONE corpus, length 2048. The horizon is an
   evaluation-time slice, and every horizon is scored on the same context
   window (see `run_eval.py --context`).

3. THE phi = 0 ROW WAS DEGENERATE. With phi = 0 the observation reduced to
   noise plus backbone, so the cross-variate knob had nothing to act on and
   the three cells of that row were byte-identical. -> The low level is 0.3,
   not 0.0: every set carries some dependency, which is what makes it a
   multivariate set at all. `--phi` restores any other choice.

=============================================================================
THE GRID
=============================================================================
Three factors, fully crossed, 27 cells:

| Axis  | Levels                        | What it controls                    |
|-------|-------------------------------|-------------------------------------|
| phi   | low 0.3 / mid 0.6 / high 0.9  | dependency ALONG TIME: the variance |
|       |                               | share of the series carried by the  |
|       |                               | context-determined factor           |
| shuf  | none / half / all             | cross-variate dependency, removed   |
|       |                               | by RE-PAIRING variates across items |
| group | full 4 / half 2 / no 1        | cross-variate dependency, removed   |
|       |                               | by SHRINKING the sharing group      |

The two cross-variate axes are deliberately different mechanisms for lowering
the same quantity, so they cross-check each other: `all` shuffle and `no`
group both drive the correlation to zero by different routes and must agree.

SHUFFLE IS A PERFECT CONTROL, WHICH IS THE POINT. The three shuffle levels of
one (phi, group) cell are derived from THE SAME generated array; a shuffle
only re-pairs which variate-series sit in one item. Every variate's marginal
law is therefore not merely equal in distribution but literally the same set
of series, so a difference between shuffle levels cannot come from the
marginal problem getting easier or harder. That is exactly what the old rho
knob could not promise: it changed `B`, hence each channel's own spectrum.

`half` shuffle re-pairs HALF THE ITEMS, not half the variates. Shuffling two
of four variates would leave one correlated pair out of six and land almost on
top of the `no`-group cell; shuffling half the items leaves half the pairs
intact and puts the midpoint where it belongs.

=============================================================================
WHY THE PERIODS ARE WHAT THEY ARE
=============================================================================
Each latent oscillator draws its period per item from {16, 32, ..., 1024}, so
an item is a mixture of fast and slow structure and the context sees a
different number of cycles for each.

The range is set against the evaluation context, not chosen for its own sake.
Measured with an oracle that grid-searches the period on the context and
extrapolates (horizon R^2, one period per sample, log-uniform in [16, 1024]):

    context     H=16    H=64    H=256    H=1024
    64          0.96    0.83    -7.2     -944
    128         0.98    0.96     0.25    -40.6
    256         0.97    0.97     0.92      0.44
    512         0.96    0.96     0.95      0.85

A negative R^2 means worse than predicting the mean: a small frequency error
estimated from a short context accumulates linearly into a phase error, and by
H=256 the extrapolation is anti-phase. Long-horizon extrapolation of an
oscillation REQUIRES a long context and no choice of period repairs that. The
default evaluation context is therefore 512, where all four horizons are
measurable and a difficulty gradient across H survives.

=============================================================================
THE GENERATIVE PROCESS
=============================================================================
For a group G of variates, with M_s shared and one private oscillator each:

    u(t)     shared bank, |G| variates see the SAME u
    v_c(t)   one private oscillator per variate
    s_c(t) = sqrt(1-beta) v_c(t) + sqrt(beta) <w, u(t)>      (LMC [1])
    x_c(t) = (1-kappa) s_c(t) + kappa g_c(s(t-L))            (CauKer [2], lag L)
    Y_c(t) = sqrt(phi) x_c(t) + sqrt(1-phi) xi_c(t)

`beta` fixes the within-group correlation and is NOT swept: the correlation
axes are `shuf` and `group`, so leaving beta free would be a third route to
the same quantity. `w` is drawn once for the build and unit-normalised, so the
shared component enters every variate of a group with the same sign — an LMC
loading matrix with free signs gives a zero-MEAN correlation whose pairwise
spread swamps it, which is the second reason the old build measured nothing.

Oscillators are written closed-form as sinusoids rather than by iterating a
rotation: the two are the same object, and the closed form does not accumulate
2048 steps of floating-point drift that a model could read as a trend.

=============================================================================
WHAT COMES OUT
=============================================================================
    <out-root>/phi{L}_sh{L}_gp{L}_s{k}/     HF DatasetDict {"train": Dataset}
                                            item_id / start / freq / target,
                                            target (V, 2048) float32
    <out-root>/metadata.json                argv, sha256 per cell, DAG edges
    <out-root>/diagnostics.json             per cell: observed correlation on
                                            the SAVED series, explained share,
                                            period mix, and the cross-channel
                                            probe (below)
    <out-root>/viz/                          one PNG per cell

THE CROSS-CHANNEL PROBE, RECORDED AND NOT ENFORCED. Correlation between
variates does not by itself make one variate's future INFERABLE from another's
past: in the old build, giving a linear oracle all eight channels instead of
one moved horizon R^2 by -0.03 to +0.03. `diagnostics.json` therefore records,
per cell, the R^2 of a ridge predicting a channel's horizon from its own
context versus from every channel's context. It is a number to read the
results against, not a gate: a flat table means something different when the
gain is 0.00 than when it is 0.20.

Usage:
    python build_dataset.py --out-root /group-volume/ts-dataset/finar_exp001
    python build_dataset.py --dry-run                    # grid + diagnostics
    python build_dataset.py --num-series 128 --shards 2  # smoke

References
----------
[1] E. Taga et al. TimePFN: Effective Multivariate Time Series Forecasting
    with Synthetic Data. AAAI 2025.  (linear coregionalization)
[2] CauKer: Classification Time Series Foundation Models Can Be Pretrained on
    Synthetic Data. ICLR 2026. arXiv:2508.02879.  (causal DAG over channels)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
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

logger = logging.getLogger("build_finar_exp001")

# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------

#: Dependency ALONG TIME: the variance share of the series carried by the
#: context-determined factor. The low level is 0.3 rather than 0.0 so that
#: every set is genuinely multivariate — at 0.0 the series is white noise and
#: neither cross-variate axis has anything to act on, which is how the previous
#: build ended up with three byte-identical cells in its lowest row.
PHI_LEVELS = {"low": 0.3, "mid": 0.6, "high": 0.9}

#: Fraction of ITEMS whose variates are re-paired across items. See the module
#: docstring: half the items, not half the variates.
SHUFFLE_LEVELS = {"none": 0.0, "half": 0.5, "all": 1.0}

#: Variates that share a latent bank. 1 makes every variate its own group,
#: which is univariate structure delivered in a multivariate container.
GROUP_SIZES = {"full": 4, "half": 2, "no": 1}

#: Drawn per item and per oscillator. Discrete rather than continuous so the
#: results can be broken down by period class for free.
PERIODS = (16, 32, 64, 128, 256, 512, 1024)

#: One length for everything. The horizon is an evaluation-time slice of this.
SERIES_LENGTH = 2048

#: Recorded for `run_eval.py`, which owns the evaluation protocol. Kept here so
#: the corpus and the protocol it was sized for stay in one place.
EVAL_HORIZONS = (16, 64, 256, 1024)
EVAL_CONTEXT = 512


def cell_name(phi: str, shuf: str, group: str) -> str:
    return f"phi{phi}_sh{shuf}_gp{group}"


def target_features():
    """The on-disk schema, shared by every writer of one of these corpora.

    EXPLICIT, because inference gets it wrong in the expensive direction:
    `Dataset.from_dict` sees a list of Python floats and infers `double`, so a
    float32 array round-trips to disk at twice its size. Measured: 16.8 MB per
    shard inferred against 8.4 MB pinned, i.e. half of the corpus and half of
    the views `run_eval.py` derives from it. `run_eval.py` imports this rather
    than restating it, so the two writers cannot drift apart.
    """
    import datasets as hf

    return hf.Features({
        "item_id": hf.Value("string"),
        "start": hf.Value("string"),
        "freq": hf.Value("string"),
        "target": hf.Sequence(hf.Sequence(hf.Value("float32"))),
    })


def groups_of(V: int, size: int) -> list[list[int]]:
    """Contiguous variate groups. ``size=1`` is V groups of one."""
    return [list(range(i, min(i + size, V))) for i in range(0, V, size)]


# --------------------------------------------------------------------------
# Latent factors
# --------------------------------------------------------------------------

def oscillator_bank(n: int, M: int, T: int, rng
                    ) -> tuple[np.ndarray, np.ndarray]:
    """``((n, T, M) oscillators, (n, M) periods)``, period drawn per item.

    sqrt(2) scales a sinusoid to unit variance, so `phi` and `beta` stay exact
    variance shares rather than approximate ones.

    Written in place: the buffer is (n, T, M) float64 — 134 MB at the default
    N — and the naive expression holds three of them at once.
    """
    t = np.arange(T, dtype=np.float64)[None, :, None]
    period = np.asarray(PERIODS, dtype=np.float64)[
        rng.integers(0, len(PERIODS), size=(n, 1, M))]
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1, M))
    buf = 2 * np.pi * t / period + phase
    np.sin(buf, out=buf)
    buf *= np.sqrt(2.0)
    return buf, period[:, 0, :]


#: CauKer's activation bank, restricted to maps that are zero-preserving and
#: 1-Lipschitz: g(0)=0 keeps the nonlinear term from injecting a level shift at
#: the seam, and |g'| <= 1 keeps kappa a share rather than a gain.
ACTIVATIONS = {
    "tanh": np.tanh,
    "sin": np.sin,
    "softsign": lambda z: z / (1.0 + np.abs(z)),
}


def sample_dag(group: list[int], p_max: int, rng
               ) -> list[tuple[int, int, str, float]]:
    """A random DAG over ONE group: ``[(child, parent, activation, weight)]``.

    Edges never cross a group boundary — the DAG is one of the mechanisms the
    `group` axis switches off, so letting it reach outside would leave a
    cross-variate pathway open in the `no`-group cells and the axis would not
    reach zero. Edges run low index to high, which is what makes it acyclic.
    """
    edges, names = [], list(ACTIVATIONS)
    for pos, child in enumerate(group):
        if pos == 0:
            continue
        k = int(rng.integers(0, min(p_max, pos) + 1))
        if k == 0:
            continue
        parents = rng.choice(group[:pos], size=k, replace=False)
        w = rng.normal(size=k)
        w /= np.abs(w).sum()
        for parent, wi in zip(np.atleast_1d(parents), np.atleast_1d(w)):
            edges.append((int(child), int(parent),
                          names[int(rng.integers(len(names)))], float(wi)))
    return edges


def couple(s: np.ndarray, edges, kappa: float, lag: int) -> np.ndarray:
    """``(1-kappa) s + kappa g(s(t-lag))``, renormalised to unit variance.

    The lag is what makes this a LEAD-LAG term: a variate's value depends on
    another variate's value at a DIFFERENT time, which is the case a single
    forward pass cannot resolve inside one column.
    """
    lagged = np.empty_like(s)
    lagged[:, :lag] = s[:, :1]
    lagged[:, lag:] = s[:, :-lag]
    out = np.zeros_like(s)
    for child, parent, act, w in edges:
        out[..., child] += w * ACTIVATIONS[act](lagged[..., parent])
    # In place from here: every one of these is (n, T, V) float64, 134 MB at
    # the default N, and `(1-kappa) * s + kappa * out` would hold two more.
    del lagged
    out *= kappa
    out += (1.0 - kappa) * s
    sd = out.std(axis=(0, 1), keepdims=True)
    out /= np.where(sd > 0, sd, 1.0)
    return out


def build_base(n: int, V: int, T: int, phi: float, beta: float, size: int,
               kappa: float, lag: int, p_max: int, rng, split: int
               ) -> tuple[np.ndarray, float, list, np.ndarray]:
    """One (phi, group) cell BEFORE shuffling — ``(Y, explained, edges, per)``.

    ``Y`` is ``(n, T, V)`` and is the observed series: there is no backbone to
    add on top of it, which is the whole point (see the module docstring).

    ``explained`` is MEASURED here rather than trusted from the algebra, and it
    comes back as the scalar it is: the deterministic part is another 134 MB
    array at the default N, and returning it would keep it alive beside ``Y``
    and the shuffled copy for the whole of the caller's inner loop.
    """
    gs = groups_of(V, size)
    n_shared = 4
    s = np.empty((n, T, V))
    per = np.empty((n, V))
    edges: list = []
    for g in gs:
        shared, _ = oscillator_bank(n, n_shared, T, rng)   # shared periods
        priv, p_pr = oscillator_bank(n, len(g), T, rng)     # are not recorded
        w = rng.normal(size=n_shared)
        w /= np.linalg.norm(w)
        common = shared @ w                                   # (n, T)
        for j, c in enumerate(g):
            s[:, :, c] = (np.sqrt(1.0 - beta) * priv[:, :, j]
                          + np.sqrt(beta) * common)
            per[:, c] = p_pr[:, j]
        edges += sample_dag(g, p_max, rng)
    x = couple(s, edges, kappa, lag)
    explained = float(phi * x[:, split:].var())
    Y = rng.normal(size=(n, T, V))
    Y *= np.sqrt(1.0 - phi)
    Y += np.sqrt(phi) * x
    return Y, explained, edges, per


def apply_shuffle(Y: np.ndarray, frac: float, rng) -> np.ndarray:
    """Re-pair variates across a fraction of the items.

    A PERMUTATION, not a resample: every variate-series that went in comes out,
    only in a different item. The marginal law of each channel is therefore
    untouched by construction rather than by argument, so a difference between
    shuffle levels can only be a difference in the JOINT structure.

    Each variate gets its OWN permutation of the selected items; using one
    permutation for all of them would move whole items around and leave every
    item's internal correlation exactly where it was.
    """
    if frac <= 0:
        return Y
    out = Y.copy()
    n = len(Y)
    k = n if frac >= 1.0 else int(round(frac * n))
    sel = rng.choice(n, size=k, replace=False)
    for c in range(Y.shape[2]):
        out[sel, :, c] = Y[rng.permutation(sel), :, c]
    return out


# --------------------------------------------------------------------------
# Diagnostics measured on what is actually written
# --------------------------------------------------------------------------

def observed_corr(Y: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    """``(signed mean, mean |corr|)`` over variate pairs of the SAVED series.

    Both, because they answer different questions and the old build was misled
    by having neither. The signed mean is the designed quantity. The absolute
    mean is the nuisance floor: variates that share a period but not a phase
    correlate by a random amount whose sign averages out, so a signed mean near
    zero beside a large absolute mean means the axis is buried in pair noise.
    """
    V = Y.shape[2]
    iu = np.triu_indices(V, 1)
    sg, ab = [], []
    for i in range(min(len(Y), 256)):
        C = np.corrcoef(Y[i, lo:hi].T)
        v = C[iu]
        v = v[np.isfinite(v)]
        if v.size:
            sg.append(v.mean())
            ab.append(np.abs(v).mean())
    return (float(np.mean(sg)) if sg else 0.0,
            float(np.mean(ab)) if ab else 0.0)


#: Ridge penalties the probe sweeps; the best is reported.
PROBE_LAMBDAS = (1e-2, 1e-1, 1.0)


def _ridge_r2(X, Y, tr, te, lams=PROBE_LAMBDAS) -> float:
    """Best held-out R^2 over ``lams``, on a split the CALLER chose.

    The split is a parameter, not drawn here, because the number this feeds is
    a COMPARISON between two designs on the same rows — drawing it per call
    would score the own-channel and all-channel models on different test sets
    and put their difference partly down to the draw.

    The Gram matrix does not depend on the penalty, so it is built once and
    only the diagonal moves across the sweep. For the all-channel design that
    is a 512x512 Gram over 1433 rows, which the naive loop rebuilt three times.
    """
    mx, my = X[tr].mean(0), Y[tr].mean(0)
    Xt = X[tr] - mx
    G, B = Xt.T @ Xt, Xt.T @ (Y[tr] - my)
    Xte, Yte = X[te] - mx, Y[te]
    den = ((Yte - Yte.mean(0)) ** 2).mean()
    if den <= 0:
        return 0.0
    eye = np.eye(X.shape[1]) * len(tr)
    best = -np.inf
    for lam in lams:
        P = Xte @ np.linalg.solve(G + lam * eye, B) + my
        best = max(best, 1.0 - ((Yte - P) ** 2).mean() / den)
    return float(best)


def cross_channel_probe(Y: np.ndarray, context: int, horizon: int, rng,
                        n_items: int = 2048, stride: int = 4) -> dict:
    """How much a channel's horizon needs the OTHER channels' context.

    Recorded, never enforced — see the module docstring. Two ridges per target
    channel, own-context and all-context, on the same split; the reported gain
    is the mean over channels. A cell where the gain is ~0 has correlation that
    a forecaster cannot convert into accuracy, and a flat row in the results
    table then says nothing about refinement.

    THE CONTEXT IS STRIDED, and that is not an optimisation. The all-channel
    design matrix has V times the columns of the own-channel one, so at full
    resolution it is the wider model that is starved of rows, and the gain
    measures conditioning rather than information — the first version of this
    probe returned a NEGATIVE gain on cells whose observed correlation is 0.66.
    A stride of 4 keeps 4 samples per cycle at the shortest period in `PERIODS`
    (Nyquist needs 2) while leaving the wide model comfortably overdetermined.
    """
    n, T, V = Y.shape
    n = min(n, n_items)
    a = T - horizon
    ctx, hor = Y[:n, a - context:a:stride, :], Y[:n, a:, :]
    # ONE split for the whole probe, and the flattened design hoisted out of
    # the comprehension — `ctx` is strided, so the reshape cannot alias and
    # copies the whole context block on every one of the V iterations.
    idx = rng.permutation(n)
    tr, te = idx[:int(0.7 * n)], idx[int(0.7 * n):]
    ctx_all = ctx.reshape(n, -1)
    own = [_ridge_r2(ctx[:, :, v], hor[:, :, v], tr, te) for v in range(V)]
    allc = [_ridge_r2(ctx_all, hor[:, :, v], tr, te) for v in range(V)]
    return {"r2_own_channel": float(np.mean(own)),
            "r2_all_channels": float(np.mean(allc)),
            "cross_channel_gain": float(np.mean(allc) - np.mean(own)),
            "probe_context": context, "probe_horizon": horizon,
            "probe_stride": stride, "probe_n_items": n,
            "probe_features_own": int(ctx.shape[1]),
            "probe_features_all": int(ctx.shape[1] * V)}


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def save_cell(Y: np.ndarray, root: Path, name: str, freq: str, start: str,
              shards: int) -> str:
    """Write one cell as ``shards`` sibling corpora and return its sha256.

    SHARDED BECAUSE THE EVALUATOR AGGREGATES: its generic yaml path emits one
    row per dataset, so a cell written as a single corpus yields n=1 and any
    per-cell dispersion statistic has no denominator. Equal-sized shards make
    the shard mean the pooled mean, so nothing else changes.
    """
    from datasets import Dataset, DatasetDict

    # `order="C"` on the astype rather than a second `ascontiguousarray` pass:
    # the transpose is not C-contiguous, so the default `order="K"` result gets
    # copied again. `sha256` reads the buffer directly, so `tobytes()` — a
    # third 67 MB copy per cell, made only to be hashed — is not needed.
    arr = Y.transpose(0, 2, 1).astype(np.float32, order="C")
    digest = hashlib.sha256(arr).hexdigest()
    n = arr.shape[0]
    bounds = np.linspace(0, n, shards + 1).astype(int)
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        if b <= a:
            continue
        part = arr[a:b]
        ds = Dataset.from_dict({
            "item_id": [f"item_{i:06d}" for i in range(a, b)],
            "start": [start] * len(part),
            "freq": [freq] * len(part),
            "target": [x.tolist() for x in part],
        }, features=target_features())
        out = root / (name if shards == 1 else f"{name}_s{k:02d}")
        out.mkdir(parents=True, exist_ok=True)
        DatasetDict({"train": ds}).save_to_disk(str(out))
    return digest


def plot_cell(Y: np.ndarray, split: int, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.0))
    lo = max(0, split - 400)
    for c in range(min(4, Y.shape[2])):
        ax.plot(range(lo, Y.shape[1]), Y[0, lo:, c], lw=0.8, label=f"var {c}")
    ax.axvline(split, color="k", ls="--", lw=0.8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, ncol=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=Path,
                   default=Path("/group-volume/ts-dataset/finar_exp001"))
    p.add_argument("--num-series", type=int, default=2048)
    p.add_argument("--num-variates", type=int, default=4)
    p.add_argument("--length", type=int, default=SERIES_LENGTH)
    p.add_argument("--phi", type=float, nargs=3, default=None,
                   metavar=("LOW", "MID", "HIGH"),
                   help="override the three dependency-along-time levels")
    p.add_argument("--beta", type=float, default=0.8,
                   help="within-group cross-variate share. NOT an axis: the "
                        "correlation axes are --shuffle and --group, and a "
                        "third route to the same quantity would confound them")
    p.add_argument("--p-max", type=int, default=2,
                   help="CauKer max-parents: within-group DAG density")
    p.add_argument("--kappa", type=float, default=0.3,
                   help="share routed through the nonlinear lagged DAG term")
    p.add_argument("--lag", type=int, default=3,
                   help="lead-lag depth of the DAG term, in steps")
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--probe-context", type=int, default=EVAL_CONTEXT)
    p.add_argument("--probe-horizon", type=int, default=256)
    p.add_argument("--no-probe", action="store_true",
                   help="skip the cross-channel probe (it is the slow part)")
    p.add_argument("--freq", default="h")
    p.add_argument("--start", default="2020-01-01T00:00:00")
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    V, T, n = args.num_variates, args.length, args.num_series
    phis = (dict(zip(("low", "mid", "high"), args.phi)) if args.phi
            else PHI_LEVELS)
    split = T - args.probe_horizon

    meta = {"argv": sys.argv,
            "config": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items()},
            "grid": {"phi": phis, "shuffle": SHUFFLE_LEVELS,
                     "group": GROUP_SIZES, "periods": list(PERIODS),
                     "eval_horizons": list(EVAL_HORIZONS),
                     "eval_context": EVAL_CONTEXT},
            "cells": {}}
    diag: dict = {"cells": {}}

    for gi, (gl, size) in enumerate(GROUP_SIZES.items()):
        for pi, (pl, phi) in enumerate(phis.items()):
            # ONE draw per (phi, group); the three shuffle levels are derived
            # from it, which is what makes the shuffle axis a pure control.
            #
            # POSITIONAL seeds, never hash(str): Python salts string hashes per
            # interpreter unless PYTHONHASHSEED is pinned, so a hash-derived
            # seed rebuilds a DIFFERENT corpus on every run. This build has
            # been bitten by exactly that before.
            base_seed = args.seed + 1000 * gi + 100 * pi
            rng = np.random.default_rng(base_seed)
            base, explained_raw, edges, per = build_base(
                n, V, T, phi, args.beta, size, args.kappa, args.lag,
                args.p_max, rng, split)
            explained = explained_raw / float(base[:, split:].var())
            for si, (sl, frac) in enumerate(SHUFFLE_LEVELS.items()):
                name = cell_name(pl, sl, gl)
                Y = apply_shuffle(base, frac,
                                   np.random.default_rng(base_seed + 10 + si))
                sg, ab = observed_corr(Y, split, T)
                cell = {
                    "phi_level": pl, "phi": phi,
                    "shuffle_level": sl, "shuffle_frac": frac,
                    "group_level": gl, "group_size": size,
                    "n_groups": len(groups_of(V, size)),
                    "beta": args.beta,
                    "observed_corr_signed": sg,
                    "observed_corr_abs": ab,
                    "measured_explained_frac": explained,
                    "series_var": float(Y.var()),
                    "period_hist_private": {str(P): int((per == P).sum())
                                    for P in PERIODS},
                }
                if not args.no_probe:
                    cell.update(cross_channel_probe(
                        Y, args.probe_context, args.probe_horizon,
                        np.random.default_rng(args.seed)))
                diag["cells"][name] = cell
                logger.info(
                    "%-26s corr=%+.3f (|.|=%.3f) explained=%.3f gain=%+.3f",
                    name, sg, ab, explained,
                    cell.get("cross_channel_gain", float("nan")))
                if args.dry_run:
                    continue
                digest = save_cell(Y, args.out_root, name, args.freq,
                                   args.start, args.shards)
                meta["cells"][name] = {"sha256": digest, "n": n, "V": V,
                                       "T": T, "dag_edges": edges}
                plot_cell(Y, split, args.out_root / "viz" / f"{name}.png",
                          f"{name}  corr={sg:+.2f}  explained={explained:.2f}")

    if args.dry_run:
        logger.info("dry run: %d cells, nothing written", len(diag["cells"]))
        return 0
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "metadata.json").write_text(json.dumps(meta, indent=1))
    (args.out_root / "diagnostics.json").write_text(json.dumps(diag, indent=1))
    logger.info("wrote %d cells x %d shards to %s",
                len(diag["cells"]), args.shards, args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
