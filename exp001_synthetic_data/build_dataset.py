#!/usr/bin/env python3
"""FiNAR Experiment 1 — a 3x3x5 grid of synthetic multivariate corpora.

Builds 45 corpora over three axes:

    cross-horizon dependency   phi in {low, mid, high}   share of the horizon
                                                         carried by a persistent
                                                         latent factor
    cross-variate dependency   rho in {low, mid, high}   how much of that factor
                                                         is SHARED across variates
    horizon length             H   in {16, 128, 256, 512, 1024}

so the FiNAR hypothesis — that iterative refinement supplies an inductive bias
for dependency WITHIN the forecast target — can be read as a surface in
(phi, rho, H) rather than a single number.

=============================================================================
THE GENERATIVE MODEL
=============================================================================
Each item i is V variates over C context steps and H horizon steps.

    latent      z(t)   = R^t z(0),   R = blockdiag(rot(theta_1..theta_M))
    signal      s(t)   = B z(t)                    (V-vector, unit variance)
    coupled     c(t)   = (1-kappa) s(t) + kappa g(s(t-L))
    residual    e(t)   = sqrt(phi) c(t) + sqrt(1-phi) xi(t)
    observed    Y(t)   = m(t) + sigma e(t)          over 0 .. C+H-1

`z` is a bank of M UNDAMPED oscillators. It is deterministic given z(0), so
z(C+k) = R^{k+1} z(C-1) — the horizon is genuinely determined by where the
context left off, and that determination does NOT decay with k.

WHY UNDAMPED, AND WHY THIS REPLACED A CONTRACTIVE VAR
The first version of this file used e(t) = Phi e(t-1) + eta(t) with Phi a
contraction of spectral radius phi. It is the natural way to write "cross-horizon
dependency", and it does not work here. For a contractive AR the share of the
horizon residual that the context seam can explain is bounded by
mean_k phi^{2(k+1)}, which was measured at:

    phi     H=16    H=128   H=1024
    0.85    0.162   0.020   0.003
    0.98    0.722   0.188   0.024
    0.999   0.983   0.881   0.425

so at H=1024 the cross-horizon axis has almost nothing to separate even at
phi=0.999, and the experiment cannot detect what it exists to measure. A
rotation has |R^k v| = |v| for every k, so the explained share is set by phi
alone and is INDEPENDENT of H — which is what makes the horizon sweep a clean
third axis instead of a confound.

The FiNAR mechanism survives the change, and is in fact sharpened by it.
Computing E[Y(C+k) | X] still requires composing the rotation k times, so a
single forward pass must produce H phase-advanced outputs jointly, while a
second pass given a plug-in estimate of Y(C+k-1) needs ONE more rotation. That
is the factorization argument in the concept note, and its difficulty grows
with H — the direction the note's own H-sweep already reports.

WHAT THE ORACLE IS — AND IS NOT
`oracle_mse = sigma^2 (1 - phi)` is the error of a forecaster that KNOWS THE
LATENT c(t) exactly. It is NOT E[(Y - E[Y|X])^2]: the context is itself observed
through noise (X carries m + sigma(sqrt(phi) c + sqrt(1-phi) xi)), so c is only
ESTIMABLE from X, never known, and the Bayes risk given X is strictly larger.

The gap is small by construction — 2048 context steps constrain 12 oscillators,
so the latent is very well identified — but it is not zero, and no model can be
expected to reach this number. It is a LOWER BOUND on achievable error, and the
excess risk computed against it is therefore an upper bound on the removable
error. Both tables label it that way; a run that appeared to beat it would mean
the bound, not the model, is wrong.

PERIOD SCALING — THE SECOND KIND OF H-INVARIANCE
There are two, and holding one does not hold the other:

  information    what share of the horizon is structured  -> `phi`, held by
                 construction and verified equal at H=16 and H=1024
  extractability how hard that structure is to USE        -> set by the
                 oscillator period RELATIVE to H, and NOT held by `phi`

With `--period-scaling absolute` the periods stay at 40-600 steps, so the
horizon spans 0.40 cycles at H=16 and 25.6 at H=1024. Estimating a frequency
from a finite context carries error, and phase error accumulates linearly in k,
so a 1% frequency error leaves corr(truth, prediction) at:

    H       P=40    P=150   P=600      P=2H (proportional)
    16      1.000   1.000   1.000      0.999
    1024    0.630   0.971   0.998      0.999

The information is all still there at H=1024 — it is simply no longer
extractable in one pass. Chronos-2 shows exactly this on the absolute arm: at
H=16 its MASE falls 0.3643 -> 0.1497 as phi rises, the intended gradient, while
at H=1024 phi=high (0.3718) is no better than phi=low (0.3699).

`--period-scaling proportional` scales the periods with H, which is what the
reference corpus does with `--ell-ratio` (ell/H held at 1, 1/3, 0) and why its
chronos-2 gradient survives to H=1000. Both arms are worth having: the
proportional one is the controlled comparison in which the H axis varies length
alone, and the absolute one is the regime where a single pass demonstrably
loses structure it could have used — which is the headroom FiNAR claims.

=============================================================================
WHERE THE CROSS-VARIATE CONSTRUCTION COMES FROM  (three sources, composed)
=============================================================================
Using one mechanism would make the result a statement about that mechanism.
Three are composed so that "cross-variate dependency" is not a synonym for
"linear Gaussian coupling":

1.  LINEAR COREGIONALIZATION — TimePFN (Taga et al., AAAI 2025), whose
    LMC-Synth draws latent GPs and mixes them into channels,
    f_d(t) = sum_q a_{d,q} u_q(t) with cross-covariance B_q = A_q A_q^T.
    `B` here is exactly that mixing matrix: variate c loads sqrt(1-rho) on a
    PRIVATE oscillator and sqrt(rho) on a bank SHARED by every variate, which
    gives corr(s_c, s_c') = rho exactly while holding Var(s_c) = 1. rho is
    therefore a knob with a closed-form meaning, not a dial that also moves the
    scale.

2.  STRUCTURAL CAUSAL EDGES — CauKer (Ke et al., ICLR 2026), which lays a random
    DAG over channels and pushes roots through nonlinear activations, with
    max-parents P_max as the density knob. `g` is that: a sparse random DAG whose
    edges apply a 1-Lipschitz, zero-preserving activation. Without it every cell
    is linear-Gaussian and the experiment measures linearity, not refinement.

3.  LEAD-LAG COUPLING — the repo's own mv_synth multivariatizer
    (`_mv_lead_lag`, `_mv_var`), whose premise is that real cross-variate
    structure is not contemporaneous. The DAG edges act at lag L > 0, so a
    variate's future depends on ANOTHER variate's future at a DIFFERENT time —
    the case a single forward pass cannot resolve within one column.

=============================================================================
WHAT IS AND IS NOT HELD FIXED
=============================================================================
HELD: the backbone m, the noise scale sigma, the total marginal variance of Y
(asserted at build time), the oscillator bank, and the item count. Across the
five horizons the context is byte-identical, so the horizon axis varies H alone.

NOT HELD: the context differs across the nine (phi, rho) cells, and it has to.
The factor must be IDENTIFIABLE from the context or there is nothing for any
forecaster to infer, and a factor visible in the context necessarily changes it.
The reference corpus (`build_cross_horizon.py`, FIRE Experiment 7) makes the
opposite choice — byte-identical context, unidentifiable future — and its
docstring records the consequence: the expected iter-k gain is then
"regime-INVARIANT, so a raw absolute-gain difference across subsets is
finite-sample noise". That design cannot test a plug-in mechanism. This one can,
at the price that cells differ in difficulty, which is why excess risk over the
per-cell oracle is the reported statistic.

Outputs, one directory per cell:

    <out-root>/
        H{H}_phi{level}_rho{level}/    HF DatasetDict {"train": Dataset} with
                                       item_id / start / freq / target, target
                                       shaped (V, C+H) — the layout every
                                       corpus in tsm-trainer already uses.
        metadata.json                  every knob, seed and a checksum per cell
        diagnostics.json               per-cell oracle, measured explained share,
                                       measured cross-variate correlation
        viz/                           one PNG per cell

Usage:
    python build_dataset.py --out-root /group-volume/ts-dataset/finar_exp001
    python build_dataset.py --dry-run                      # grid + diagnostics
    python build_dataset.py --horizons 16 128 --num-series 256   # smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("build_finar_exp001")

# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------

#: Cross-horizon knob: the VARIANCE SHARE of the horizon residual carried by the
#: persistent factor. 0.0 makes the residual pure white noise, so E[e(C+k)|X]=0
#: and there is nothing along the time axis for a second pass to exploit — the
#: honest floor. 0.9 leaves a tenth of the residual irreducible, so even the
#: strongest cell has a finite oracle and a model cannot score 0.
PHI_LEVELS = {"low": 0.0, "mid": 0.5, "high": 0.9}

#: Cross-variate knob: the correlation between two variates' factor components.
#: 0.0 leaves V independent univariate problems sharing only the noise scale;
#: 0.9 makes nine tenths of each variate's structure inferable from the others.
RHO_LEVELS = {"low": 0.0, "mid": 0.5, "high": 0.9}

#: The horizon sweep. 16 and 128 sit inside the 75-patch training range of the
#: EO v4 configs; 256/512/1024 walk out toward the 1200-step ceiling that
#: max_prediction_patch=75 x input_patch_size=16 imposes, so the longest cell is
#: near the architectural limit and not past it.
HORIZONS = (16, 128, 256, 512, 1024)


def cell_name(H: int, phi: str, rho: str) -> str:
    return f"H{H}_phi{phi}_rho{rho}"


# --------------------------------------------------------------------------
# Latent factors and the cross-variate mixing
# --------------------------------------------------------------------------

def oscillator_bank(n: int, M: int, T: int, rng, p_lo: int, p_hi: int
                    ) -> np.ndarray:
    """(n, T, M) unit-variance undamped oscillators, one phase set per item.

    Written as sinusoids rather than by iterating a rotation matrix: they are
    the same object (z(t) = R^t z(0) with R a rotation IS a sinusoid) and the
    closed form avoids accumulating T matrix products of floating-point error
    over 3072 steps, which at H=1024 would show up as a slow amplitude drift
    that a model could read as a trend.

    sqrt(2) scales each to unit variance, so `phi` is a variance share exactly.
    """
    t = np.arange(T, dtype=np.float64)[None, :, None]
    period = rng.uniform(p_lo, p_hi, size=(n, 1, M))
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1, M))
    return np.sqrt(2.0) * np.sin(2 * np.pi * t / period + phase)


def shared_weights(n_shared: int, rng) -> np.ndarray:
    """The shared bank's loading direction, drawn ONCE for the whole build.

    Drawn once and passed in, not redrawn inside `mixing_matrix`, because
    `calibrate_rho` calls that function tens of times: a fresh draw per call
    makes the bisection optimise a MOVING TARGET, and the value it converges on
    then describes a mixing matrix that generation never uses.

    The effect is second-order rather than a bias — every variate loads on the
    SAME w and ||w|| = 1, so corr(s_c, s_c') = rho exactly whatever direction w
    points, and only the interaction with the nonlinear DAG moves. That is
    precisely the part the calibration exists to correct, so it should not be
    resampled underneath it.
    """
    w = rng.normal(size=n_shared)
    return w / np.linalg.norm(w)


def mixing_matrix(V: int, rho: float, n_shared: int, w: np.ndarray) -> np.ndarray:
    """(V, V + n_shared) LMC loading: private oscillators plus a shared bank.

    Variate c takes sqrt(1-rho) of its OWN oscillator and sqrt(rho) of a shared
    combination. With unit-variance independent oscillators this gives

        Var(s_c)          = (1-rho) + rho = 1
        Cov(s_c, s_c')    = rho                     for c != c'

    exactly, so rho is the cross-variate correlation and nothing else moves.
    The shared weights are normalised to unit L2 for the same reason.
    """
    B = np.zeros((V, V + n_shared))
    B[np.arange(V), np.arange(V)] = np.sqrt(1.0 - rho)
    if rho > 0:
        B[:, V:] = np.sqrt(rho) * w[None, :]
    return B


#: CauKer's activation bank, restricted to maps that are both zero-preserving
#: and 1-Lipschitz. g(0)=0 keeps the nonlinear term from injecting a constant
#: drift (a level shift at the seam would be trivially detectable); |g'| <= 1
#: bounds its contribution so kappa stays a share rather than a gain. A cubic
#: was in this bank and was removed: clipped at +-3 its slope reaches 3.
ACTIVATIONS = {
    "tanh": np.tanh,
    "sin": np.sin,
    "softsign": lambda z: z / (1.0 + np.abs(z)),
}


def sample_dag(V: int, p_max: int, rng) -> list[tuple[int, int, str, float]]:
    """A random DAG over variates: ``[(child, parent, activation, weight)]``.

    CauKer's P_max is the density knob. Edges run from lower to higher index,
    which is what makes it acyclic; the variate order is arbitrary so this
    costs no generality. Each child's incoming weights are normalised to
    sum|w| = 1, so a child that drew two parents does not thereby get twice the
    gain of one that drew a single parent.
    """
    edges, names = [], list(ACTIVATIONS)
    for child in range(1, V):
        k = int(rng.integers(0, min(p_max, child) + 1))
        if k == 0:
            continue
        parents = rng.choice(child, size=k, replace=False)
        w = rng.normal(size=k)
        w /= np.abs(w).sum()
        for parent, wi in zip(parents, w):
            edges.append((child, int(parent),
                          names[int(rng.integers(len(names)))], float(wi)))
    return edges


def apply_dag(s_lag: np.ndarray, edges, V: int) -> np.ndarray:
    out = np.zeros_like(s_lag)
    for child, parent, act, w in edges:
        out[..., child] += w * ACTIVATIONS[act](s_lag[..., parent])
    return out


def couple(s: np.ndarray, edges, kappa: float, lag: int, V: int) -> np.ndarray:
    """(1-kappa) s(t) + kappa g(s(t-L)), renormalised to unit variance.

    Renormalised because the DAG output is not unit-variance (the activations
    squash, and a child with no parents contributes nothing), and without the
    rescale `phi` would stop being a variance share the moment kappa > 0 —
    silently coupling the nonlinearity knob to the cross-horizon axis.
    """
    if not kappa or not edges:
        return s
    lagged = np.concatenate([np.zeros_like(s[:, :lag, :]), s[:, :-lag, :]], axis=1)
    c = (1.0 - kappa) * s + kappa * apply_dag(lagged, edges, V)
    sd = c.std(axis=(0, 1), keepdims=True)
    return c / np.where(sd > 0, sd, 1.0)


def measured_corr(osc, B, edges, kappa, lag, V) -> float:
    """Mean off-diagonal correlation of the coupled signal, as generated."""
    c = couple(osc @ B.T, edges, kappa, lag, V)
    cc = np.corrcoef(c.reshape(-1, V).T)
    return float((cc.sum() - V) / (V * (V - 1)))


def calibrate_rho(target: float, V: int, n_shared: int, edges, kappa: float,
                  lag: int, osc: np.ndarray, w: np.ndarray, tol: float = 5e-3,
                  iters: int = 40) -> tuple[float, float]:
    """Loading rho_b whose GENERATED cross-variate correlation is ``target``.

    The nominal loading does not survive `couple`: the DAG term is itself a
    cross-variate mechanism, so it both dilutes a requested correlation and
    contributes one of its own. Measured against nominal, uncalibrated:

        rho = 0.0  ->  -0.035      (the DAG's own coupling, not zero)
        rho = 0.5  ->   0.384      (diluted by a fifth)
        rho = 0.9  ->   0.847

    Reporting those as "low / mid / high = 0.0 / 0.5 / 0.9" would put a number
    in the paper that the data does not have, and the mid cell would sit closer
    to 0.4 than to the midpoint it is supposed to mark. Bisection on the
    loading fixes both, and the achieved value is recorded either way.

    Bisects on rho_b in [0, 1); monotone because raising the shared loading can
    only raise the shared variance share. Returns ``(rho_b, achieved)``.
    """
    lo, hi = 0.0, 0.999
    best = (0.0, measured_corr(osc, mixing_matrix(V, 0.0, n_shared, w),
                               edges, kappa, lag, V))
    if abs(best[1] - target) < tol:
        return best
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        got = measured_corr(osc, mixing_matrix(V, mid, n_shared, w),
                            edges, kappa, lag, V)
        if abs(got - target) < abs(best[1] - target):
            best = (mid, got)
        if abs(got - target) < tol:
            break
        if got < target:
            lo = mid
        else:
            hi = mid
    return best


def backbone(n: int, V: int, T: int, periods, rng) -> np.ndarray:
    """(n, T, V) deterministic seasonal mean, one phase/amplitude per variate."""
    t = np.arange(T, dtype=np.float64)[None, :, None]
    m = rng.normal(0.0, 0.5, size=(n, 1, V))
    for P in periods:
        amp = rng.uniform(0.5, 1.5, size=(n, 1, V))
        psi = rng.uniform(0, 2 * np.pi, size=(n, 1, V))
        m = m + amp * np.sin(2 * np.pi * t / P + psi)
    return m


def build_cell(m_full, osc, white, phi, rho_b, edges, kappa, lag, sigma, V, w
               ) -> tuple[np.ndarray, np.ndarray]:
    """``(target, signal)`` for one cell — (n, T, V) each.

    ``osc`` and ``white`` are SHARED across the nine cells AT A GIVEN HORIZON,
    so two cells of one horizon differ only through B(rho) and the phi mix.
    They are NOT shared between horizons — see the module docstring. ``signal`` is the
    deterministic part of the residual and is returned so the caller can
    measure the oracle and the realised cross-variate correlation instead of
    trusting the algebra.
    """
    B = mixing_matrix(V, rho_b, n_shared=osc.shape[2] - V, w=w)
    s = osc @ B.T
    c = couple(s, edges, kappa, lag, V)
    e = np.sqrt(phi) * c + np.sqrt(1.0 - phi) * white
    return m_full + sigma * e, np.sqrt(phi) * c


def save_cell(target: np.ndarray, root: Path, name: str, freq: str, start: str,
              shards: int) -> str:
    """Write one cell as ``shards`` sibling corpora and return its sha256.

    SHARDED BECAUSE THE EVALUATOR AGGREGATES. tsm-trainer's generic yaml path
    emits ONE ROW PER DATASET, so a cell written as a single corpus of 2048
    series yields n=1 and a per-cell win rate degenerates to 0 or 1 — measured,
    not assumed: the first smoke run returned exactly that. Sixteen shards of
    128 series give a win rate over 16 units, and because the shards are equal
    sized their mean is the pooled mean, so no other statistic changes.

    It is affordable because the loader has no fixed per-dataset cost: loading
    2048 series took 2.01s and 128 took 0.10s, i.e. the cost is in the bytes,
    and the same bytes split 16 ways cost the same.

    The digest is of the WHOLE cell, before splitting, so the byte-identity
    check across cells is unaffected by the shard count.
    """
    from datasets import Dataset, DatasetDict

    arr = np.ascontiguousarray(target.transpose(0, 2, 1).astype(np.float32))
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
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
        })
        out = root / (name if shards == 1 else f"{name}_s{k:02d}")
        out.mkdir(parents=True, exist_ok=True)
        DatasetDict({"train": ds}).save_to_disk(str(out))
    return digest


def plot_cell(target: np.ndarray, C: int, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.0))
    lo = max(0, C - 200)
    for c in range(min(4, target.shape[2])):
        ax.plot(range(lo, target.shape[1]), target[0, lo:, c], lw=0.8,
                label=f"var {c}")
    ax.axvline(C, color="k", ls="--", lw=0.8)
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
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    p.add_argument("--num-series", type=int, default=2048)
    p.add_argument("--num-variates", type=int, default=8)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--num-shared", type=int, default=4,
                   help="oscillators in the SHARED bank (the rho pathway)")
    p.add_argument("--period-scaling", choices=("absolute", "proportional"),
                   default="absolute",
                   help="ABSOLUTE holds the oscillator periods fixed in steps, "
                        "so a longer horizon spans more cycles and the same "
                        "structure gets harder to extrapolate. PROPORTIONAL "
                        "scales them with H, so the horizon always spans the "
                        "same number of cycles — the choice the reference "
                        "cross-horizon corpus makes with --ell-ratio. See "
                        "PERIOD SCALING in the module docstring.")
    p.add_argument("--period-range", type=float, nargs=2, default=[40.0, 600.0],
                   help="oscillator period range in STEPS, when "
                        "--period-scaling absolute")
    p.add_argument("--period-mult", type=float, nargs=2, default=[0.25, 2.0],
                   help="oscillator period range as MULTIPLES OF H, when "
                        "--period-scaling proportional. The default spans "
                        "0.5-4 cycles per horizon at every H, and keeps at "
                        "least one full period inside a 2048-step context "
                        "even at H=1024.")
    p.add_argument("--periods", type=int, nargs="+", default=[24, 96, 168],
                   help="backbone seasonalities")
    p.add_argument("--sigma", type=float, default=0.5,
                   help="scale of the whole horizon residual")
    p.add_argument("--p-max", type=int, default=2,
                   help="CauKer max-parents: cross-variate DAG density")
    p.add_argument("--kappa", type=float, default=0.3,
                   help="share of the signal routed through the nonlinear "
                        "lagged DAG term; 0 makes every cell linear-Gaussian")
    p.add_argument("--lag", type=int, default=3,
                   help="lead-lag depth of the DAG term, in steps")
    p.add_argument("--shards", type=int, default=16,
                   help="sibling corpora per cell. The evaluator reports one "
                        "row per dataset, so this is what gives a per-cell win "
                        "rate a denominator; 1 restores a single corpus.")
    p.add_argument("--freq", default="h")
    p.add_argument("--start", default="2020-01-01T00:00:00")
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--dry-run", action="store_true",
                   help="report the grid and its diagnostics without writing")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    V, C, n = args.num_variates, args.context_length, args.num_series
    M = V + args.num_shared

    dag_rng = np.random.default_rng(args.seed)
    edges = sample_dag(V, args.p_max, dag_rng)
    # ONE draw for the whole build, so calibration and generation describe the
    # same mixing matrix — see shared_weights.
    w_shared = shared_weights(args.num_shared, dag_rng)
    logger.info("cross-variate DAG: %d edges over %d variates (P_max=%d)",
                len(edges), V, args.p_max)

    meta = {"argv": sys.argv,
            "config": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items()},
            "dag_edges": edges, "cells": {}}
    diag: dict = {"cells": {}}

    for H in args.horizons:
        T = C + H
        h_rng = np.random.default_rng(args.seed + H)
        m_full = backbone(n, V, T, args.periods, h_rng)
        p_lo, p_hi = (args.period_range if args.period_scaling == "absolute"
                      else [m * H for m in args.period_mult])
        logger.info("  H=%-5d period range %.1f-%.1f steps (%.2f-%.2f cycles "
                    "per horizon)", H, p_lo, p_hi, H / p_hi, H / p_lo)
        osc = oscillator_bank(n, M, T, h_rng, p_lo, p_hi)
        white = h_rng.standard_normal((n, T, V))

        # Calibrate on a SUBSAMPLE: the correlation is a population quantity
        # and 128 items over the shared bank estimate it to well inside the
        # 5e-3 tolerance, while calibrating on all 2048 would cost 40 full
        # couples per level.
        cal_osc = osc[: min(128, n)]
        rho_b = {}
        for rl, rho in RHO_LEVELS.items():
            b, got = calibrate_rho(rho, V, args.num_shared, edges, args.kappa,
                                   args.lag, cal_osc, w_shared)
            rho_b[rl] = b
            logger.info("  H=%-5d rho %-5s target %.2f -> loading %.4f "
                        "(achieved %.3f)", H, rl, rho, b, got)

        for pl, phi in PHI_LEVELS.items():
            for rl, rho in RHO_LEVELS.items():
                name = cell_name(H, pl, rl)
                # One rng per cell, seeded positionally — str hashing is salted
                # per interpreter unless PYTHONHASHSEED is pinned, so a hashed
                # seed would rebuild a different corpus on every run.
                target, sig = build_cell(m_full, osc, white, phi, rho_b[rl],
                                         edges, args.kappa, args.lag,
                                         args.sigma, V, w_shared)

                fut = slice(C, T)
                # MEASURED, not assumed. The oracle knows the deterministic part
                # exactly, so its residual is the white component: sigma^2(1-phi).
                resid = (target - m_full) / args.sigma
                explained = float(sig[:, fut].var() / resid[:, fut].var())
                oracle = float(args.sigma**2 * (1.0 - phi))
                # Realised cross-variate correlation of the structured part,
                # averaged over variate pairs — this is what rho promises.
                f = sig[:, fut].reshape(-1, V)
                cc = np.corrcoef(f.T) if phi > 0 else np.eye(V)
                xvar = float((cc.sum() - V) / (V * (V - 1)))

                diag["cells"][name] = {
                    "H": H, "phi_level": pl, "phi": phi,
                    "rho_level": rl, "rho": rho, "rho_loading": rho_b[rl],
                    "oracle_mse_latent_lower_bound": oracle,
                    "period_scaling": args.period_scaling,
                    "period_lo": p_lo, "period_hi": p_hi,
                    "cycles_per_horizon": [H / p_hi, H / p_lo],
                    "measured_explained_frac": explained,
                    "measured_cross_variate_corr": xvar,
                    "horizon_var": float(target[:, fut].var()),
                    "context_var": float(target[:, :C].var()),
                }
                if args.dry_run:
                    logger.info(
                        "  [dry] %-24s oracle=%.4f explained=%.3f (phi=%.2f) "
                        "xcorr=%.3f (rho=%.2f) var=%.3f",
                        name, oracle, explained, phi, xvar, rho,
                        diag["cells"][name]["horizon_var"])
                    continue

                digest = save_cell(target, args.out_root, name, args.freq,
                                   args.start, args.shards)
                meta["cells"][name] = {"sha256": digest, "n": n, "V": V,
                                       "C": C, "H": H, "shards": args.shards}
                plot_cell(target, C, args.out_root / "viz" / f"{name}.png", name)
                logger.info("  %-24s oracle=%.4f explained=%.3f sha=%s",
                            name, oracle, explained, digest[:12])

    if args.dry_run:
        logger.info("dry run: %d cells", len(diag["cells"]))
        print(json.dumps(diag, indent=1)[:0])  # keep json import honest
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "metadata.json").write_text(json.dumps(meta, indent=1))
    (args.out_root / "diagnostics.json").write_text(json.dumps(diag, indent=1))
    logger.info("wrote %d cells to %s", len(meta["cells"]), args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
