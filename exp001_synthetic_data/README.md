# Appendix: Synthetic Data for the Dependency Ablation (FiNAR Experiment 1)

This appendix documents the construction of the synthetic corpora used to test
whether iterative refinement supplies an inductive bias for dependency *within*
the forecast target. Every number quoted is reproduced by
`python build_dataset.py --dry-run`.

## A.1 Design

We construct a $3 \times 3 \times 3$ grid of corpora over three factors of
variation, giving 27 cells:

| Axis | Levels | Meaning |
|---|---|---|
| $\phi$ | 0.3 / 0.6 / 0.9 | dependency **along time**: the variance share of the series carried by the context-determined factor |
| shuffle | none / half / all | dependency **across variates**, removed by re-pairing variates across items |
| group | 4 / 2 / 1 | dependency **across variates**, removed by shrinking the group that shares a latent bank |

Each cell holds $N = 2048$ items of $V = 4$ variates over $T = 2048$ steps,
written as 8 equal shards so that every reported statistic has a dispersion
estimate. The forecast horizon is not an axis of the corpus: it is an
evaluation-time slice (A.4).

$\phi$ is the share of the observation that is predictable in principle; the
complementary share is white noise. The two cross-variate axes are two
independent mechanisms for lowering the same quantity, so they cross-check each
other — `all` shuffle and `no` group reach zero correlation by different routes
and must agree.

## A.2 Generative process

For a group $G$ of variates, with $M_s = 4$ shared oscillators and one private
oscillator per variate:

$$
\begin{aligned}
u(t)   &\quad \text{shared bank; every variate of } G \text{ sees the same } u \\
v_c(t) &\quad \text{one private oscillator per variate} \\
s_c(t) &= \sqrt{1-\beta}\, v_c(t) + \sqrt{\beta}\, \langle w, u(t)\rangle
          &&\text{(linear coregionalization [1])} \\
x_c(t) &= (1-\kappa)\, s_c(t) + \kappa\, g_c\big(s(t - L)\big)
          &&\text{(causal DAG over channels [2], lead-lag } L) \\
Y_c(t) &= \sqrt{\phi}\, x_c(t) + \sqrt{1-\phi}\, \xi_c(t),
          \qquad \xi \sim \mathcal{N}(0, I)
\end{aligned}
$$

with $\beta = 0.8$, $\kappa = 0.3$, $L = 3$ and max-parents $P_{\max} = 2$.
Activations are drawn from $\{\tanh, \sin, \mathrm{softsign}\}$, all
zero-preserving and 1-Lipschitz, so the nonlinear term neither injects a level
shift nor turns $\kappa$ from a share into a gain.

$Y$ is the whole observation; nothing is added on top of it, so the structured
component carries the entire variance and the designed correlation is the
correlation of the series that is written.

$\beta$ is fixed rather than swept: the cross-variate axes are `shuffle` and
`group`, and a third route to the same quantity would confound them. $w$ is
drawn once per group and unit-normalised, so the shared component enters every
variate of a group with the same sign.

**Periods.** Each oscillator draws its period per item from
$\{16, 32, 64, 128, 256, 512, 1024\}$, so an item mixes fast and slow structure
and the context sees a different number of cycles for each. Oscillators are
written closed-form as sinusoids rather than by iterating a rotation matrix: the
two are the same object, and the closed form does not accumulate 2048 steps of
floating-point drift that a model could read as a trend.

## A.3 The two cross-variate mechanisms

**Shuffle** re-pairs which variate-series sit inside one item, as a permutation.
The three shuffle levels of a $(\phi, \text{group})$ cell are derived from the
*same* generated array, so every variate's marginal law is not merely equal in
distribution but literally the same set of series. A difference between shuffle
levels therefore cannot come from the marginal problem getting easier or harder;
only the joint structure changes. `half` re-pairs half the **items**, which
leaves half the variate pairs intact — with $V = 4$ there are six pairs, and
shuffling two of four *variates* would instead leave only one.

**Group** sets how many variates share a latent bank: 4 (one group), 2 (two
groups) or 1 (four groups, i.e. univariate structure delivered in a multivariate
container). The causal DAG edges never cross a group boundary, so that this axis
closes every cross-variate pathway rather than all but one.

## A.4 Evaluation protocol

The corpus is built once at $T = 2048$. A horizon
$H \in \{16, 64, 256, 1024\}$ is a **view**,
`target[..., anchor - C : anchor + H]`, with the anchor fixed at
$2048 - 1024 = 1024$ and the context $C = 512$ by default. Two properties
follow: a trend down the $H$ axis compares one corpus with itself rather than
four independent draws, and every horizon is scored on the identical context
window.

Fixing the context matters because the evaluator's context is otherwise
everything before the split, i.e. $2048 - H$, which would shrink by a factor of
64 across this sweep and vary horizon length and context length together.

**Why 512 steps.** An oracle that grid-searches the oscillator period on the
context and extrapolates reaches, in horizon $R^2$:

| context | $H{=}16$ | $H{=}64$ | $H{=}256$ | $H{=}1024$ |
|---|---|---|---|---|
| 64 | 0.96 | 0.83 | −7.2 | −944 |
| 128 | 0.98 | 0.96 | 0.25 | −40.6 |
| 256 | 0.97 | 0.97 | 0.92 | 0.44 |
| 512 | 0.96 | 0.96 | 0.95 | 0.85 |

A negative $R^2$ is worse than predicting the mean: a frequency error estimated
from a short context accumulates linearly into a phase error, and by $H = 256$
the extrapolation is anti-phase. 512 is the shortest context at which all four
horizons are measurable and a difficulty gradient across $H$ still survives.

## A.5 Verification

Diagnostics are measured on the series that is written, not on the latent
components. Observed cross-variate correlation, averaged over variate pairs, at
$N = 2048$:

| $\phi$ | shuffle none | half | all | | group 4 | 2 | 1 |
|---|---|---|---|---|---|---|---|
| 0.3 | +0.201 | +0.112 | +0.004 | | +0.201 | +0.069 | +0.000 |
| 0.6 | +0.387 | +0.188 | +0.010 | | +0.387 | +0.144 | −0.005 |
| 0.9 | +0.646 | +0.353 | +0.001 | | +0.646 | +0.241 | −0.005 |

(shuffle at group = 4; group at shuffle = none, so the two share their first
column.) The shuffle ladder is $\{1, \tfrac12, 0\}$ and the group ladder is
$\{1, \tfrac13, 0\}$ of the intact value, as the mechanisms predict, and the two
agree at zero. The measured variance share is 0.298 / 0.602 / 0.900 against
targets of 0.3 / 0.6 / 0.9.

The two axes are not orthogonal in the observed correlation and cannot be:
correlation is carried by the structured component and the complementary noise
is independent per variate, so $\phi$ sets the ceiling
($\operatorname{corr} \approx 0.72\,\phi$ at group 4) and the cross-variate axes
lower the correlation from it. Comparisons of a cross-variate effect are
therefore read **within** a $\phi$ row.

`diagnostics.json` also records, per cell, the horizon $R^2$ of a ridge
predicting a channel from its own context versus from every channel's context.
At $\phi = 0.9$ and group 4 the gain is $-0.005 / -0.011 / -0.012$ across
shuffle none / half / all: it orders with the correlation but is approximately
zero, because a channel's own 512-step history already identifies its own signal.
Cross-variate information in these corpora is redundant rather than necessary,
and the number is recorded so that a flat result can be read against it.

## A.6 Files

| File | Purpose |
|---|---|
| `build_dataset.py` | builds the 27 corpora, `metadata.json`, `diagnostics.json`, `viz/` |
| `run_eval.py` | materialises the context+$H$ views and scores one model at every recursion depth; imports tsm-trainer, never modifies it |
| `run_exp001.sh` | end-to-end driver, with the remote-ssh usage in its header |
| `build_table.py` | the two tables: iteration 1 vs 2, improvement, win rate |

Evaluation imports `Evaluator.evaluate_benchmark(config_path=...)` directly
rather than going through `run_evaluation.sh`, because `run_benchmark.py`
resolves benchmark configs from a hard-coded directory inside tsm-trainer and
routing an external corpus through it would require writing a file there.

## References

[1] E. Taga et al. *TimePFN: Effective Multivariate Time Series Forecasting
with Synthetic Data.* AAAI 2025.

[2] CauKer: *Classification Time Series Foundation Models Can Be Pretrained on
Synthetic Data.* ICLR 2026. arXiv:2508.02879.
