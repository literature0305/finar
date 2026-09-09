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

## A.7 Sub-experiment 1-2: cross-horizon dependency x horizon length

A second, separately-built corpus asks whether the refinement gain tracks
dependency *across the horizon*, and how that changes as the horizon grows.
Twelve tasks — 4 horizons $\{16, 100, 400, 1000\}$ x 3 dependency regimes
$\{$high, mid, low$\}$ — scored at every recursion depth $1..N$.

**Why the published corpus could not be used directly.** The source
(`/group-volume/ts-dataset/cross_horizon_length`) holds the observed context
byte-identical across all twelve subsets, 8192 series x 2048 steps. The *model*
does not see 2048 of them: `aed/model_base.py::append_forecast_region` takes the
forecast region out of the same window,

$$\text{max\_ctx} = \text{context\_length} - \lceil H/P \rceil \cdot P$$

so at $\text{context\_length}=2048$, $P=16$ the history reaching the encoder is
2032 / 1936 / 1648 / **1040** for $H = 16/100/400/1000$. The horizon axis is then
collinear with "how much history the model got", and a MASE that rises with $H$
cannot be attributed to the horizon.

`build_cross_horizon_trim.py` re-cuts every subset to **1040** observed steps —
what the longest horizon leaves — so no horizon truncates. 1040 and not 1048:
$\lceil 1000/16 \rceil \cdot 16 = 1008$, not 1000. The twelve still share a
byte-identical observed window, and `start` is advanced by the 1008 dropped
steps so timestamps still describe the data.

*Not* preserved: the source README's oracle and Chronos-2 MASE, measured against
2048 steps and a denominator computed from them. Numbers here are comparable to
each other, not to that table.

**Depth beyond training is the question, not an accident.** `--iters` may exceed
`coe_train_depth_max`; `EOPipeline.report_depth` is `max(trained, requested)`,
so a $K=2$ checkpoint asked for 4 reports `repeat1..repeat4`. Those rows are
recorded and flagged (`beyond_trained_depth` in the csv, a red line in the
figure) rather than hidden.

```bash
bash run_exp001-2_cross-horizon.sh --ckpt /path/to/eo-v4/best_checkpoints --iters 4
```

Full options are in the script header (`--help`). Files:
`build_cross_horizon_trim.py`, `run_eval_cross_horizon.py`,
`build_table_cross_horizon.py`, `run_exp001-2_cross-horizon.sh`.

### Observed history, both corpora

| $H$ | $\lceil H/P \rceil \cdot P$ | published: stored / effective | trimmed: stored / effective |
|---:|---:|---:|---:|
| 16 | 16 | 2048 / 2032 | 1040 / 1040 |
| 100 | 112 | 2048 / 1936 | 1040 / 1040 |
| 400 | 400 | 2048 / 1648 | 1040 / 1040 |
| 1000 | 1008 | 2048 / **1040** | 1040 / 1040 |

### Verification run (eo-v4-120M-toto_2080ti, $K=2$, depths 1–4)

MASE, all twelve tasks:

| $H$ | regime | iter1 | iter2 | iter3 | iter4 |
|---:|---|---:|---:|---:|---:|
| 16 | high | 0.8942 | **0.8898** | 0.9091 | 0.9312 |
| 16 | mid | 1.1677 | **1.1648** | 1.1808 | 1.1992 |
| 16 | low | 1.2969 | **1.2918** | 1.3050 | 1.3210 |
| 100 | high | 0.9954 | **0.9715** | 0.9875 | 1.0069 |
| 100 | mid | 1.2360 | **1.2151** | 1.2274 | 1.2437 |
| 100 | low | 1.3918 | **1.3712** | 1.3812 | 1.3954 |
| 400 | high | 1.0394 | **1.0286** | 1.0560 | 1.0936 |
| 400 | mid | 1.2716 | **1.2621** | 1.2853 | 1.3176 |
| 400 | low | 1.4272 | **1.4181** | 1.4394 | 1.4694 |
| 1000 | high | **1.1186** | 1.1511 | 1.2485 | 1.3730 |
| 1000 | mid | **1.3455** | 1.3744 | 1.4606 | 1.5731 |
| 1000 | low | **1.4920** | 1.5208 | 1.6031 | 1.7118 |

Three things this shows, on this checkpoint:

1. **The dependency gradient survives the re-cut.** high $<$ mid $<$ low at
   every horizon and every depth, so the regime still means what it meant.
2. **Iteration 2 is the optimum wherever the model was trained for it** — at
   $H \le 400$ it beats iteration 1 in all nine cells — and depths 3–4, which
   this checkpoint never saw, degrade monotonically.
3. **$H = 1000$ inverts it:** iteration 1 is best and every further pass hurts,
   by up to $-22.7\%$. With the observed history now held equal, that is not a
   context-length artefact. It is the one cell where the refinement is
   counterproductive, and it is also the horizon the checkpoint saw least —
   `prediction_patch_alpha = 1.5` weights $p(k) \propto k^{-1.5}$, so $H=1000$
   (63 patches) drew about 0.08% of training samples against 42% for $H=16$.
   Whether the inversion is the horizon or the training scarcity is not settled
   by this run.

## References

[1] E. Taga et al. *TimePFN: Effective Multivariate Time Series Forecasting
with Synthetic Data.* AAAI 2025.

[2] CauKer: *Classification Time Series Foundation Models Can Be Pretrained on
Synthetic Data.* ICLR 2026. arXiv:2508.02879.
