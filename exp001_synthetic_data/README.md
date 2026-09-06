# Appendix: Synthetic Data for the Dependency Ablation (FiNAR Experiment 1)

This appendix documents the construction of the synthetic corpora used to test
whether iterative refinement supplies an inductive bias for dependency *within*
the forecast target. It is written to be dropped into a paper appendix; every
number quoted is reproduced by `python build_dataset.py --dry-run`.

## A.1 Design

We construct a $3 \times 3 \times 5$ grid of corpora over three axes:

| Axis | Symbol | Levels | Meaning |
|---|---|---|---|
| Cross-horizon dependency | $\phi$ | 0.0 / 0.5 / 0.9 | variance share of the horizon carried by a persistent latent factor |
| Cross-variate dependency | $\rho$ | 0.0 / 0.5 / 0.9 | correlation between two variates' factor components |
| Horizon length | $H$ | 16 / 128 / 256 / 512 / 1024 | forecast length |

giving 45 corpora. Each holds $N = 2048$ items of $V = 8$ variates over
$C = 2048$ context steps followed by $H$ horizon steps.

## A.2 Generative process

For item $i$ and variate $c$, with $t = 0, \dots, C + H - 1$:

$$
\begin{aligned}
z(t) &= R^{t} z(0), \qquad R = \mathrm{blockdiag}\big(\mathrm{rot}(\theta_1), \dots, \mathrm{rot}(\theta_M)\big) \\
s(t) &= B\, z(t) \\
c(t) &= (1-\kappa)\, s(t) + \kappa\, g\big(s(t - L)\big) \\
e(t) &= \sqrt{\phi}\, c(t) + \sqrt{1-\phi}\, \xi(t), \qquad \xi(t) \sim \mathcal{N}(0, I) \\
Y(t) &= m(t) + \sigma\, e(t)
\end{aligned}
$$

with $m$ a per-variate sum of three sinusoids (periods 24 / 96 / 168, random
phase and amplitude), $\sigma = 0.5$, $M = V + 4 = 12$ oscillators with periods
drawn uniformly from $[40, 600]$ steps, $\kappa = 0.3$ and $L = 3$.

$z$ is a bank of **undamped** oscillators, so it is deterministic given
$z(0)$ and $z(C+k) = R^{k+1} z(C-1)$: the horizon is determined by where the
context left off, and that determination does not decay with $k$.

**The oracle.** Because $c$ is deterministic given the context and $\xi$ is
white, $\mathbb{E}[Y(C+k) \mid X] = m(C+k) + \sigma \sqrt{\phi}\, c(C+k)$, so the
irreducible per-step MSE is exactly

$$\mathrm{MSE}^\star = \sigma^2 (1 - \phi)$$

at every horizon: 0.250 / 0.125 / 0.025 for $\phi \in \{0, 0.5, 0.9\}$.

## A.3 Why undamped oscillators rather than a contractive VAR

The natural way to write "cross-horizon dependency" is a stable VAR,
$e(t) = \Phi e(t-1) + \eta(t)$ with $\|\Phi\| = \phi < 1$. We built that first
and discarded it. For a contractive AR, the share of the horizon residual that
the context seam can explain is bounded by $\frac{1}{H}\sum_{k<H} \phi^{2(k+1)}$:

| $\phi$ | $H{=}16$ | $H{=}128$ | $H{=}1024$ |
|---|---|---|---|
| 0.85 | 0.162 | 0.020 | 0.003 |
| 0.98 | 0.722 | 0.188 | 0.024 |
| 0.999 | 0.983 | 0.881 | 0.425 |

At $H = 1024$ the cross-horizon axis has almost nothing left to separate, even
at $\phi = 0.999$ — the horizon length would silently destroy the very
dependency the grid varies. A rotation satisfies $\|R^k v\| = \|v\|$ for all
$k$, so the explained share is set by $\phi$ alone and is independent of $H$,
which is what makes the horizon a clean third axis instead of a confound.

The refinement mechanism survives the change and is sharpened by it. Computing
$\mathbb{E}[Y(C+k) \mid X]$ still requires composing the rotation $k$ times, so a
single forward pass must emit $H$ phase-advanced outputs jointly, whereas a
second pass given a plug-in estimate of $Y(C+k-1)$ needs one further rotation.
The difficulty of the one-pass problem therefore grows with $H$.

## A.4 Cross-variate structure: three composed mechanisms

Using a single mechanism would make the result a statement about that
mechanism, so $\rho$ acts through three, composed:

1. **Linear coregionalization**, after TimePFN's LMC-Synth [1], which draws
   latent GPs and mixes them into channels, $f_d(t) = \sum_q a_{d,q} u_q(t)$
   with cross-covariance $B_q = A_q A_q^{\top}$. Here $B$ gives variate $c$ a
   loading $\sqrt{1-\rho}$ on a private oscillator and $\sqrt{\rho}$ on a bank
   shared by all variates, so $\operatorname{Var}(s_c) = 1$ and
   $\operatorname{Cov}(s_c, s_{c'}) = \rho$.
2. **Structural causal edges**, after CauKer [2], which lays a random DAG over
   channels and pushes roots through nonlinear activations with max-parents
   $P_{\max}$ as the density knob. $g$ is a sparse DAG ($P_{\max} = 2$) whose
   edges apply a 1-Lipschitz, zero-preserving activation drawn from
   $\{\tanh, \sin, \mathrm{softsign}\}$. Without it every cell is
   linear-Gaussian and the experiment measures linearity, not refinement.
3. **Lead-lag coupling**, after the multivariatizer's `_mv_lead_lag` /
   `_mv_var`. DAG edges act at lag $L = 3$, so a variate's future depends on
   another variate's future *at a different time* — the case a single forward
   pass cannot resolve within one column.

**Calibration.** The nominal loading does not survive the DAG, which is itself
a cross-variate mechanism: it both dilutes a requested correlation and
contributes one of its own. Measured against nominal, uncalibrated:
$\rho = 0.0 \mapsto -0.035$, $0.5 \mapsto 0.384$, $0.9 \mapsto 0.847$. Reporting
those as $\{0, 0.5, 0.9\}$ would put a number in the paper the data does not
have, and the middle cell would sit nearer 0.4 than the midpoint it marks. We
therefore bisect on the loading until the *generated* correlation matches the
target, achieving 0.003 / 0.503 / 0.901 at $H{=}16$ and 0.004 / 0.499 / 0.897 at
$H{=}1024$. The achieved value is recorded per cell in `diagnostics.json`.

## A.5 What is and is not held fixed

**Held within a horizon.** The backbone $m$, the noise scale $\sigma$, the
oscillator bank, the item count, and the total marginal variance of $Y$
(2.06–2.15 across all 45 cells).

**Not held across horizons.** The per-horizon generator is seeded $\texttt{seed}+H$,
so the five horizons draw independent contexts from the same law rather than
sharing one (context SHA-256 of $H{=}16/128/1024$: `b6ac7ee4`, `a161c74b`,
`bd9986d4`). A MASE denominator is therefore equal across horizons only in
expectation, and reading one $\phi$ level down the $H$ axis is a distributional
rather than an exact comparison. The shard-to-shard standard deviation is
0.002–0.008 MASE, which bounds how much of any $H$ trend this can explain.

**Not held.** The context differs across the nine $(\phi, \rho)$ cells, and it
must: the factor has to be identifiable from the context or there is nothing
for any forecaster to infer, and a factor visible in the context necessarily
changes it.

This is the opposite of the choice made by the cross-horizon benchmark of the
companion FIRE experiment, which holds the context byte-identical and varies
only the future noise law. That design has a property its own documentation
records: with $\mathbb{E}[Y \mid X]$ equal across regimes, the expected iteration-$k$
gain is regime-invariant, so a raw absolute-gain difference between regimes is
finite-sample noise. Such a design cannot test a plug-in mechanism. Ours can, at
the price that cells differ in difficulty — which is why the reported
cross-cell statistic is excess risk over the per-cell oracle,

$$\text{closed} = \frac{(\mathrm{MSE}_1 - \mathrm{MSE}^\star) - (\mathrm{MSE}_2 - \mathrm{MSE}^\star)}{\mathrm{MSE}_1 - \mathrm{MSE}^\star},$$

a share of the *removable* error. Raw MASE may be read down a column but never
across one.

## A.6 Verification

`build_dataset.py --dry-run` reports, per cell, the analytic oracle, the
measured explained share and the measured cross-variate correlation. At
$N = 256$:

| | measured explained ($\phi$) | measured corr ($\rho$) |
|---|---|---|
| target 0.0 / 0.5 / 0.9 | 0.000 / 0.502 / 0.901 ($H{=}16$) | 0.003 / 0.503 / 0.901 |
| target 0.0 / 0.5 / 0.9 | 0.000 / 0.499 / 0.899 ($H{=}1024$) | 0.004 / 0.499 / 0.897 |

The explained share is identical at $H{=}16$ and $H{=}1024$, confirming the
horizon axis is orthogonal to the dependency axes.

Two defects found and fixed during construction are recorded here because they
would each have produced a plausible-looking but wrong grid:

- An earlier version added the nonlinear DAG term *beside* the AR term rather
  than inside the contraction, giving loop gain $\phi(1+\kappa) > 1$. Horizon
  variance diverged to 648 at $(\phi{=}0.98, \rho{=}0)$ against ~3.1 at the same
  $\phi$ with $\rho > 0$, because raising $\rho$ lowered $V-1$ of $\Phi$'s
  eigenvalues and quietly damped the blow-up. The instability therefore tracked
  the *other* axis.
- Cell seeds were derived from `hash((str, str))`, which Python salts per
  interpreter unless `PYTHONHASHSEED` is pinned, so the corpus would have been
  different on every rebuild. Seeds are now positional.

## A.7 Files

| File | Purpose |
|---|---|
| `build_dataset.py` | builds the 45 corpora, `metadata.json`, `diagnostics.json`, `viz/` |
| `run_eval.py` | scores one model at every recursion depth; imports tsm-trainer, never modifies it |
| `run_exp001.sh` | end-to-end driver, with the remote-ssh usage in its header |
| `build_table.py` | iteration-1 vs iteration-2 tables: MASE, WQL, win rate, excess risk |

Evaluation imports `Evaluator.evaluate_benchmark(config_path=...)` directly
rather than going through `run_evaluation.sh`, because `run_benchmark.py`
resolves benchmark configs from a hard-coded directory inside tsm-trainer and
routing an external corpus through it would require writing a file there.

## References

[1] E. Taga et al. *TimePFN: Effective Multivariate Time Series Forecasting
with Synthetic Data.* AAAI 2025.

[2] CauKer: *Classification Time Series Foundation Models Can Be Pretrained on
Synthetic Data.* ICLR 2026. arXiv:2508.02879.
